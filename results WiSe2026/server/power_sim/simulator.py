from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .profiles import load_profile


@dataclass
class Estimate:
    model_id: str
    energy_mj: Dict[str, float]
    total_energy_mj: float
    avg_power_mw: float
    battery_wh_default: float
    estimated_runtime_hours: Optional[float]
    throttled: bool
    breakdown_mw: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "energy_mj": self.energy_mj,
            "total_energy_mj": self.total_energy_mj,
            "avg_power_mw": self.avg_power_mw,
            "battery_wh_default": self.battery_wh_default,
            "estimated_runtime_hours": self.estimated_runtime_hours,
            "throttled": self.throttled,
            "breakdown_mw": self.breakdown_mw,
        }


def _tier_from_util(u: float) -> str:
    if u <= 0.34:
        return "low"
    if u <= 0.67:
        return "mid"
    return "high"


def _power_mw(domain: Dict[str, Any], util: float) -> float:
    util = max(0.0, min(1.0, float(util)))
    idle = float(domain.get("idle_mw", 0.0))
    active_tbl = domain.get("active_mw_at_util1", {}) or {}
    tier = _tier_from_util(util)
    if active_tbl:
        active = float(active_tbl.get(tier, max(active_tbl.values())))
    else:
        active = idle
    return idle + util * max(0.0, active - idle)


def _estimate_runtime_hours(battery_wh: float, avg_power_mw: float) -> Optional[float]:
    # runtime = Wh / W; power in mW
    if avg_power_mw <= 1e-9:
        return None
    return float(battery_wh) / (float(avg_power_mw) / 1000.0)


def _apply_thermal_cap(avg_power_mw: float, thermal: Dict[str, Any]) -> Tuple[float, bool]:
    sustained = float(thermal.get("sustained_mw", avg_power_mw))
    if avg_power_mw > sustained:
        return sustained, True
    return avg_power_mw, False


def _normalize_mix(mix: Dict[str, float]) -> Dict[str, float]:
    """Ensure weights sum to 1.0 (robust to rounding / user mistakes)."""
    clean = {k: float(v) for k, v in mix.items() if v is not None and float(v) > 0.0}
    s = sum(clean.values())
    if s <= 1e-12:
        return {"cpu_eff": 1.0}
    return {k: v / s for k, v in clean.items()}


def _default_stage_mix(mode: str, policy: str) -> Dict[str, Dict[str, float]]:
    """
    Returns a per-stage mixture of domains (weights sum to ~1.0).
    Keys are domain names that must exist in profiles.json domains:
      - cpu_eff, cpu_perf, gpu, npu, memory
    """
    cpu_dom = "cpu_perf" if policy == "performance" else "cpu_eff" if policy == "battery" else "cpu_eff"

    decode_mix = {cpu_dom: 0.85, "memory": 0.15}
    post_mix = {cpu_dom: 0.80, "memory": 0.20}

    if mode == "object":
        if policy == "performance":
            infer_mix = {"gpu": 0.60, cpu_dom: 0.30, "memory": 0.10}
        elif policy == "battery":
            infer_mix = {"npu": 0.80, cpu_dom: 0.15, "memory": 0.05}
        else:  # balanced
            infer_mix = {"npu": 0.70, cpu_dom: 0.20, "memory": 0.10}

    elif mode == "face":
        if policy == "performance":
            infer_mix = {"gpu": 0.50, cpu_dom: 0.40, "memory": 0.10}
        elif policy == "battery":
            infer_mix = {"npu": 0.35, cpu_dom: 0.55, "memory": 0.10}
        else:  # balanced
            infer_mix = {"npu": 0.30, cpu_dom: 0.60, "memory": 0.10}

    else:  # scene
        if policy == "performance":
            infer_mix = {"npu": 0.60, cpu_dom: 0.30, "memory": 0.10}
        elif policy == "battery":
            infer_mix = {"npu": 0.45, cpu_dom: 0.45, "memory": 0.10}
        else:  # balanced
            infer_mix = {"npu": 0.50, cpu_dom: 0.40, "memory": 0.10}

    return {
        "decode": _normalize_mix(decode_mix),
        "inference": _normalize_mix(infer_mix),
        "post": _normalize_mix(post_mix),
    }


def estimate_from_timings(
    model_id: str,
    timings_ms: Dict[str, float],
    mode: str,
    policy: str = "balanced",
    extra: Optional[Dict[str, Any]] = None,
) -> Estimate:
    """
    Estimate energy from measured pipeline stage timings.
    timings_ms example: {"decode":4.2, "inference":32.1, "post":6.5}
    Returns energy per stage (mJ), total mJ, avg mW, runtime estimate.
    """
    profile = load_profile(model_id)
    scales = profile.get("scales", {}) or {}
    domains = profile.get("domains", {}) or {}
    thermal = profile.get("thermal", {}) or {}
    battery_wh = float(profile.get("battery_wh_default", 19.0))

    budgets = {"object": 33.0, "face": 33.0, "scene": 200.0}
    budget_ms = float(budgets.get(mode, 50.0))

    include_overhead = bool(extra.get("include_overhead")) if extra else True
    overhead_mw = float(extra.get("overhead_mw", 250.0)) if extra else 250.0

    # Optionally allow overriding the stage mix from server
    stage_mix = _default_stage_mix(mode, policy)
    if extra and isinstance(extra.get("stage_mix"), dict):
        # Expect: {"decode": {"cpu_eff":0.8,"memory":0.2}, "inference": {...}, "post": {...}}
        try:
            for stg, mix in extra["stage_mix"].items():
                if isinstance(mix, dict):
                    stage_mix[stg] = _normalize_mix(mix)
        except Exception:
            # If user sends a bad mix, just ignore and use defaults
            pass

    # Collect per-stage energy
    energy_mj: Dict[str, float] = {}
    total_energy_mj = 0.0
    total_time_s = 0.0

    # Domain-level energy accumulation (for breakdown)
    domain_energy_mj: Dict[str, float] = {
        "cpu": 0.0,
        "gpu": 0.0,
        "npu": 0.0,
        "memory": 0.0,
        "overhead": 0.0,
    }
    domain_time_s = 0.0

    # Only consider known stages; ignore "total" if client sends it
    for stage, ms in (timings_ms or {}).items():
        if stage not in ("decode", "inference", "post"):
            continue
        if ms is None:
            continue

        ms_f = max(0.0, float(ms))
        dt_s = ms_f / 1000.0
        total_time_s += dt_s
        domain_time_s += dt_s

        # Utilization proxy: stage_ms relative to per-mode budget
        util = min(1.0, ms_f / max(1e-6, budget_ms))

        mix = stage_mix.get(stage, {"cpu_eff": 1.0})
        mix = _normalize_mix(mix)

        # Weighted stage power from multiple domains
        stage_p_mw = 0.0

        for dom_name, w in mix.items():
            dom = domains.get(dom_name, {})
            p = _power_mw(dom, util)

            # Apply scale depending on domain
            if dom_name.startswith("cpu"):
                p *= float(scales.get("cpu", 1.0))
                domain_energy_mj["cpu"] += (p * float(w) * dt_s)
            elif dom_name == "gpu":
                p *= float(scales.get("gpu", 1.0))
                domain_energy_mj["gpu"] += (p * float(w) * dt_s)
            elif dom_name == "npu":
                p *= float(scales.get("npu", 1.0))
                domain_energy_mj["npu"] += (p * float(w) * dt_s)
            elif dom_name == "memory":
                domain_energy_mj["memory"] += (p * float(w) * dt_s)

            stage_p_mw += float(w) * p

        # Overhead is applied during active work; scaled by util
        oh_mw = (overhead_mw * util) if include_overhead else 0.0
        domain_energy_mj["overhead"] += oh_mw * dt_s

        # Energy: mW * s = mJ
        e_mj = (stage_p_mw + oh_mw) * dt_s
        energy_mj[stage] = round(e_mj, 4)
        total_energy_mj += e_mj

    avg_power_mw = (total_energy_mj / total_time_s) if total_time_s > 1e-9 else 0.0
    avg_power_mw_capped, throttled = _apply_thermal_cap(avg_power_mw, thermal)

    # If capped, scale energies proportionally (simple proxy)
    if throttled and avg_power_mw > 1e-9:
        scale = avg_power_mw_capped / avg_power_mw
        for k in list(energy_mj.keys()):
            energy_mj[k] = round(energy_mj[k] * scale, 4)
        for k in list(domain_energy_mj.keys()):
            domain_energy_mj[k] *= scale
        total_energy_mj *= scale
        avg_power_mw = avg_power_mw_capped

    runtime_h = _estimate_runtime_hours(battery_wh, avg_power_mw)

    breakdown_mw: Dict[str, float] = {}
    if domain_time_s > 1e-9:
        breakdown_mw = {k: round(v / domain_time_s, 2) for k, v in domain_energy_mj.items()}

    return Estimate(
        model_id=model_id,
        energy_mj=energy_mj,
        total_energy_mj=round(total_energy_mj, 4),
        avg_power_mw=round(avg_power_mw, 2),
        battery_wh_default=battery_wh,
        estimated_runtime_hours=(round(runtime_h, 2) if runtime_h is not None else None),
        throttled=throttled,
        breakdown_mw=breakdown_mw,
    )


def estimate_from_workload(
    model_id: str,
    fps: float,
    util_cpu: float,
    util_gpu: float = 0.0,
    util_npu: float = 0.0,
    policy: str = "balanced",
    battery_wh: Optional[float] = None,
) -> Estimate:
    """
    Steady-state estimate (no camera needed). Energy_mj is "per second" here.
    """
    profile = load_profile(model_id)
    scales = profile.get("scales", {}) or {}
    domains = profile.get("domains", {}) or {}
    thermal = profile.get("thermal", {}) or {}
    battery_wh = float(battery_wh or profile.get("battery_wh_default", 19.0))

    cpu_domain_name = "cpu_perf" if policy == "performance" else "cpu_eff"
    cpu_domain = domains.get(cpu_domain_name, {})
    gpu_domain = domains.get("gpu", {})
    npu_domain = domains.get("npu", {})
    mem_domain = domains.get("memory", {})

    p_cpu = _power_mw(cpu_domain, util_cpu) * float(scales.get("cpu", 1.0))
    p_gpu = _power_mw(gpu_domain, util_gpu) * float(scales.get("gpu", 1.0))
    p_npu = _power_mw(npu_domain, util_npu) * float(scales.get("npu", 1.0))
    p_mem = _power_mw(mem_domain, max(util_cpu, util_gpu, util_npu) * 0.5) if mem_domain else 0.0

    avg_power_mw = p_cpu + p_gpu + p_npu + p_mem
    avg_power_mw, throttled = _apply_thermal_cap(avg_power_mw, thermal)

    energy_mj = {
        "cpu": round(p_cpu, 3),
        "gpu": round(p_gpu, 3),
        "npu": round(p_npu, 3),
        "memory": round(p_mem, 3),
        "per_second_total": round(avg_power_mw, 3),
        "fps": float(fps),
    }

    runtime_h = _estimate_runtime_hours(battery_wh, avg_power_mw)

    return Estimate(
        model_id=model_id,
        energy_mj=energy_mj,
        total_energy_mj=round(avg_power_mw, 3),
        avg_power_mw=round(avg_power_mw, 2),
        battery_wh_default=battery_wh,
        estimated_runtime_hours=(round(runtime_h, 2) if runtime_h is not None else None),
        throttled=throttled,
        breakdown_mw={
            "cpu": round(p_cpu, 2),
            "gpu": round(p_gpu, 2),
            "npu": round(p_npu, 2),
            "memory": round(p_mem, 2),
        },
    )
