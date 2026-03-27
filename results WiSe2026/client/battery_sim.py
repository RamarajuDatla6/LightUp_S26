# battery_sim.py

import psutil
import requests
import time
from collections import deque

TAB_CPU_SERVER = "http://localhost:5050/get_cpu"

# -----------------------------------------------------
# GLOBAL STATE FOR TRACKING HISTORY
# -----------------------------------------------------
class CPUTracker:
    def __init__(self, history_size=30):
        """Track CPU usage over time for better predictions"""
        self.cpu_history = deque(maxlen=history_size)  # Last 30 readings
        self.timestamp_history = deque(maxlen=history_size)
        
    def add_sample(self, cpu_value):
        """Add a CPU measurement"""
        if cpu_value is not None:
            self.cpu_history.append(cpu_value)
            self.timestamp_history.append(time.time())
    
    def get_average_cpu(self, window_seconds=10):
        """Get average CPU over recent time window"""
        if len(self.cpu_history) == 0:
            return None
        
        now = time.time()
        recent_samples = []
        
        for i in range(len(self.cpu_history) - 1, -1, -1):
            if now - self.timestamp_history[i] <= window_seconds:
                recent_samples.append(self.cpu_history[i])
            else:
                break
        
        if recent_samples:
            return sum(recent_samples) / len(recent_samples)
        return None
    
    def get_trend(self):
        """Detect if CPU usage is increasing, stable, or decreasing"""
        if len(self.cpu_history) < 5:
            return "stable"
        
        recent = list(self.cpu_history)[-5:]
        older = list(self.cpu_history)[-10:-5] if len(self.cpu_history) >= 10 else recent
        
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        
        diff = recent_avg - older_avg
        
        if diff > 5:
            return "increasing"
        elif diff < -5:
            return "decreasing"
        else:
            return "stable"

# Global tracker instance
cpu_tracker = CPUTracker()

# -----------------------------------------------------
# GET REAL BATTERY %
# -----------------------------------------------------
def get_battery_percent():
    try:
        bat = psutil.sensors_battery()
        if bat:
            return bat.percent
    except:
        pass
    return None


# -----------------------------------------------------
# GET TAB CPU FROM CHROME EXTENSION SERVER
# -----------------------------------------------------
def get_tab_cpu():
    try:
        r = requests.get(TAB_CPU_SERVER, timeout=1)
        if r.status_code == 200:
            data = r.json()
            return data.get("cpu")
    except:
        pass
    return None


# -----------------------------------------------------
# GET TOTAL SYSTEM CPU (RECOMMENDED)
# -----------------------------------------------------
def get_system_cpu():
    """
    Get overall system CPU usage.
    This captures ALL processes including your backend server.
    """
    try:
        # interval=0.5 gives accurate reading without blocking too long
        cpu_percent = psutil.cpu_percent(interval=0.5, percpu=False)
        return cpu_percent
    except:
        pass
    return None


# -----------------------------------------------------
# GET ALL CPU CORES USAGE (OPTIONAL - FOR DETAILED VIEW)
# -----------------------------------------------------
def get_per_core_cpu():
    """Get CPU usage per core (useful for multi-core systems)"""
    try:
        per_cpu = psutil.cpu_percent(interval=0.5, percpu=True)
        return per_cpu
    except:
        pass
    return None


# -----------------------------------------------------
# ADVANCED BATTERY DRAIN ALGORITHM
# -----------------------------------------------------
def estimate_runtime_advanced(battery_percent, current_cpu, avg_cpu, cpu_trend):
    """
    Advanced battery runtime estimation algorithm
    
    Algorithm factors:
    1. Current battery level
    2. Current CPU usage
    3. Average CPU usage (smoothed)
    4. CPU trend (increasing/stable/decreasing)
    5. Base battery capacity assumption
    6. Non-linear drain patterns
    
    Returns: estimated hours remaining
    """
    if battery_percent is None or current_cpu is None:
        return None
    
    # Use average CPU if available, otherwise current
    effective_cpu = avg_cpu if avg_cpu is not None else current_cpu
    
    # --- BASE BATTERY MODEL ---
    # Assume laptop has ~10 hours at idle (5-10% CPU)
    # and ~2-3 hours at full load (100% CPU)
    BASE_IDLE_HOURS = 10.0
    BASE_FULL_LOAD_HOURS = 2.5
    
    # --- CPU TO DRAIN RATE MAPPING ---
    # Non-linear relationship: higher CPU = exponentially more drain
    cpu_normalized = max(min(effective_cpu, 100), 0) / 100.0
    
    # Power consumption curve (exponential-like)
    # Low CPU (0-20%): ~1x drain
    # Medium CPU (20-50%): ~2-3x drain  
    # High CPU (50-80%): ~4-6x drain
    # Very high CPU (80-100%): ~7-10x drain
    if cpu_normalized < 0.2:
        drain_multiplier = 1.0 + (cpu_normalized * 2)
    elif cpu_normalized < 0.5:
        drain_multiplier = 1.4 + (cpu_normalized * 3)
    elif cpu_normalized < 0.8:
        drain_multiplier = 2.9 + (cpu_normalized * 4)
    else:
        drain_multiplier = 6.1 + (cpu_normalized * 4)
    
    # --- TREND ADJUSTMENT ---
    # If CPU is increasing, assume higher future drain
    # If decreasing, assume lower future drain
    trend_adjustment = 1.0
    if cpu_trend == "increasing":
        trend_adjustment = 1.15  # 15% pessimistic adjustment
    elif cpu_trend == "decreasing":
        trend_adjustment = 0.90  # 10% optimistic adjustment
    
    drain_multiplier *= trend_adjustment
    
    # --- CALCULATE RUNTIME ---
    # Base hours scaled by battery level and drain rate
    base_hours_at_current_load = BASE_IDLE_HOURS / drain_multiplier
    remaining_hours = (battery_percent / 100.0) * base_hours_at_current_load
    
    # --- SAFETY BOUNDS ---
    # Don't predict impossibly long or short times
    max_possible = BASE_IDLE_HOURS * (battery_percent / 100.0)
    min_possible = BASE_FULL_LOAD_HOURS * (battery_percent / 100.0)
    
    remaining_hours = max(min_possible, min(remaining_hours, max_possible))
    
    return round(remaining_hours, 2)


# -----------------------------------------------------
# SIMPLE RUNTIME ESTIMATION (FALLBACK)
# -----------------------------------------------------
def estimate_runtime_simple(battery_percent, cpu_percent):
    """
    Simple estimation (original method, kept as fallback)
    """
    if battery_percent is None or cpu_percent is None:
        return None

    base_hours = 10.0
    cpu_factor = max(cpu_percent, 1) / 100.0
    drain_multiplier = 1 + (cpu_factor * 2)
    remaining_hours = (battery_percent / 100.0) * base_hours / drain_multiplier

    return round(remaining_hours, 2)


# -----------------------------------------------------
# MAIN FUNCTION USED BY STREAMLIT
# -----------------------------------------------------
def get_system_stats():
    tab_cpu = get_tab_cpu()
    system_cpu = get_system_cpu()  # Total system CPU
    battery = get_battery_percent()
    
    # Track CPU history
    if system_cpu is not None:
        cpu_tracker.add_sample(system_cpu)
    
    # Get smoothed metrics
    avg_cpu = cpu_tracker.get_average_cpu(window_seconds=10)
    cpu_trend = cpu_tracker.get_trend()
    
    # Calculate runtime using advanced algorithm
    runtime = estimate_runtime_advanced(
        battery_percent=battery,
        current_cpu=system_cpu,
        avg_cpu=avg_cpu,
        cpu_trend=cpu_trend
    )
    
    # Fallback to simple if advanced fails
    if runtime is None and system_cpu is not None:
        runtime = estimate_runtime_simple(battery, system_cpu)
    
    return {
        "battery_percent": battery,
        "browser_cpu": tab_cpu,
        "system_cpu": system_cpu,  # Changed from python_cpu
        "avg_cpu": avg_cpu,  # New: smoothed average
        "cpu_trend": cpu_trend,  # New: trend indicator
        "estimated_runtime_h": runtime
    }


# -----------------------------------------------------
# DETAILED SYSTEM INFO (OPTIONAL)
# -----------------------------------------------------
def get_detailed_stats():
    """Get comprehensive system statistics"""
    battery = get_battery_percent()
    tab_cpu = get_tab_cpu()
    system_cpu = get_system_cpu()
    per_core = get_per_core_cpu()
    
    # Memory info
    memory = psutil.virtual_memory()
    
    # Disk info
    disk = psutil.disk_usage('/')
    
    return {
        "battery_percent": battery,
        "browser_cpu": tab_cpu,
        "system_cpu": system_cpu,
        "cpu_per_core": per_core,
        "memory_percent": memory.percent,
        "memory_available_gb": round(memory.available / (1024**3), 2),
        "disk_percent": disk.percent,
        "disk_free_gb": round(disk.free / (1024**3), 2)
    }


# -----------------------------------------------------
# TEST FUNCTION
# -----------------------------------------------------
if __name__ == "__main__":
    """
    Test the battery estimation algorithm
    """
    import time
    print("=" * 60)
    print("BATTERY & CPU MONITORING TEST")
    print("=" * 60)
    print("\nRun your face recognition to see CPU increase!\n")
    
    while True:
        stats = get_system_stats()
        
        print(f"🔋 Battery: {stats['battery_percent']}%")
        print(f"🌐 Browser Tab CPU: {stats['browser_cpu']}%")
        print(f"💻 System CPU (Current): {stats['system_cpu']}%")
        print(f"📊 System CPU (10s Avg): {stats['avg_cpu']}%")
        print(f"📈 CPU Trend: {stats['cpu_trend']}")
        print(f"⏱️  Estimated Runtime: {stats['estimated_runtime_h']} hours")
        print("-" * 60)
        
        time.sleep(2)