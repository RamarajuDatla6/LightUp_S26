"""Power simulation package.

This module adds a lightweight, explainable power/energy simulation for modern
mobile SoCs.

It is designed for a *project/demo* setting:

* Uses per-domain (CPU perf/eff, GPU, NPU) power tables per SoC.
* Estimates energy from measured pipeline timings (ms) + a simple utilization model.
* Exposes FastAPI endpoints so the Streamlit client can demonstrate results on a laptop.

Notes on validity
-----------------
Public sources rarely expose exact vendor power tables. The included profiles are
*starting points* and intentionally include calibration factors.

For a "valid" simulation in a student project, calibrate each SoC profile with
at least one real measurement run (Android power profiler / Xcode Instruments)
and document the calibration in your report.
"""

from .profiles import list_models, load_profile
from .simulator import estimate_from_timings, estimate_from_workload

__all__ = [
    "list_models",
    "load_profile",
    "estimate_from_timings",
    "estimate_from_workload",
]
