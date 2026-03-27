import random
import psutil
import math

def smart_battery_drain(usage_rate, battery_percentage, time_interval):
    drain = usage_rate * time_interval
    return max(battery_percentage - drain, 0)

def exponential_battery_drain(battery_percentage, lambda_value, time_interval):
    drain = battery_percentage * (1 - math.exp(-lambda_value * time_interval))
    return max(battery_percentage - drain, 0)

def cpu_based_battery_drain(cpu_usage, battery_percentage, time_interval, drain_factor=0.02):
    drain = (cpu_usage / 100) * time_interval * drain_factor
    return max(battery_percentage - drain, 0)

def temperature_based_battery_drain(temperature, battery_percentage, time_interval, temperature_factor=0.01):
    drain = (temperature / 100) * time_interval * temperature_factor
    return max(battery_percentage - drain, 0)

def random_event_drain(battery_percentage):
    event_factor = random.choice([0, 0.01, 0.02])
    return max(battery_percentage - battery_percentage * event_factor, 0) 