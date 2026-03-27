import platform
import psutil
import subprocess
import time
import os
try:    
    import wmi
except:
    wmi = None 
# def get_real_battery():
#     bat = psutil.sensors_battery()
#     return bat

# # battery = get_real_battery()
# # print(battery)
# running = []
# def detect_running_browser():
#     browser_map = {"Chrome":"chrome.exe","Edge":"msedge.exe","Firefox": "firefox.exe"}
    
#     for p in psutil.process_iter(["name"]):
#         for browser,exe in browser_map.items():
#             if p.info["name"] == exe and browser not in running:
#                 running.append(browser)

# detect_running_browser()
# print(running)

# def get_os():
    # return platform.system()

# def get_real_temperature():
#     os_name = get_os()

#     if os_name == "Windows":
#         if wmi is None:
            
#             return None
#         try:
#             w = wmi.WMI(namespace="root\\OpenHardwareMonitor")
#             sensors = w.Sensor()
#             temps = [s.Value for s in sensors if s.SensorType == "Temperature"]
#             return temps[0] if temps else None
#         except:
#             return None
# print(get_real_temperature())

# import psutil
# import time

# # ---- MODEL PARAMETERS (tunable) ----
# BASE_TEMP = 38.0        # idle temperature (Dell i7 realistic)
# MAX_TEMP = 95.0         # safe upper bound
# HEAT_RATE = 0.12        # heating speed
# COOL_RATE = 0.06        # cooling speed

# _last_temp = BASE_TEMP
# _last_time = time.time()

import subprocess
import os
import re

def get_battery_capacity_wh():
    home = os.path.expanduser("~")
    report_path = os.path.join(home, "battery-report.html")

    # Generate report
    subprocess.run(
        ["powercfg", "/batteryreport", "/output", report_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    if not os.path.exists(report_path):
        return None

    with open(report_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Normalize text (important!)
    text = text.replace(",", "").lower()

    # Try full charge capacity first
    full_match = re.search(r"full charge capacity.*?(\d{4,6})", text)
    if full_match:
        return round(int(full_match.group(1)) / 1000, 1)

    # Fallback: design capacity
    design_match = re.search(r"design capacity.*?(\d{4,6})", text)
    if design_match:
        return round(int(design_match.group(1)) / 1000, 1)

    return None

battery_capacity = get_battery_capacity_wh()

import psutil
import time

def get_chrome_cpu_percent():
    chrome_processes = []

    # First pass: find chrome processes and prime CPU counters
    for p in psutil.process_iter(['name']):
        if p.info['name'] == 'chrome.exe':
            try:
                p.cpu_percent(None)
                chrome_processes.append(p)
            except:
                pass

    # wait to measure real CPU usage
    time.sleep(0.3)

    cpu_total = 0.0
    for p in chrome_processes:
        try:
            cpu_total += p.cpu_percent(None)
        except:
            pass

    return round(cpu_total, 1)
chrome_cpu_percent = get_chrome_cpu_percent()

def estimate_chrome_runtime_hours(battery_wh, chrome_power_w):
    if not battery_wh or not chrome_power_w or chrome_power_w <= 0:
        return None

    runtime_hours = battery_wh / chrome_power_w
    return round(runtime_hours, 2)
print("Battery (Wh):", battery_capacity)
print("Chrome Power (W):", chrome_cpu_percent)
print("Estimated Runtime (hours):",
estimate_chrome_runtime_hours(battery_capacity, chrome_cpu_percent))