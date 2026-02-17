#!/usr/bin/env python3
"""
VIOLET BAND: Thermal Monitoring System
Framework: Gamesa Cortex V2 / Neural Compositor
Module: resource_governance

This module provides real-time thermal telemetry to the Economic Governor.
It attempts to read system sensors (via `sensors` command) and falls back
to a simulation model if unavailable.

Usage:
    monitor = ThermalMonitor()
    temp = monitor.get_temperature()
    penalty = monitor.calculate_thermal_penalty()
"""

import subprocess
import re
import random
import time
import logging

class ThermalMonitor:
    def __init__(self, critical_temp=85.0, throttle_temp=75.0):
        self.logger = logging.getLogger("ThermalMonitor")
        self.critical_temp = critical_temp
        self.throttle_temp = throttle_temp
        self.simulated_temp = 45.0
        self.trend = 0.5 # Heating up
        self.sensors_available = self._check_sensors()

    def _check_sensors(self):
        """Check if `sensors` command is available."""
        try:
            subprocess.run(["sensors"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except FileNotFoundError:
            self.logger.warning("`sensors` command not found. Running in Simulation Mode.")
            return False

    def get_temperature(self):
        """Get current CPU/GPU temperature."""
        if self.sensors_available:
            try:
                output = subprocess.check_output(["sensors"]).decode("utf-8")
                # Regex to find Core/Package temps (standard format)
                match = re.search(r"Package id 0:\s+\+([0-9.]+)", output)
                if match:
                    return float(match.group(1))
                # Fallback to Core 0
                match = re.search(r"Core 0:\s+\+([0-9.]+)", output)
                if match:
                    return float(match.group(1))
            except Exception:
                pass # Fallback to simulation
        
        # Simulation Logic
        self.simulated_temp += self.trend * random.uniform(0.1, 0.5)
        if self.simulated_temp > 80: self.trend = -0.5 # Cooling
        if self.simulated_temp < 40: self.trend = 0.5 # Heating
        return self.simulated_temp

    def calculate_thermal_penalty(self):
        """
        Returns a penalty factor (0.0 to 1.0).
        0.0 = No Penalty
        1.0 = Full Throttle (Halting)
        """
        temp = self.get_temperature()
        
        if temp < self.throttle_temp:
            return 0.0
        elif temp >= self.critical_temp:
            return 1.0
        else:
            # Linear ramp between throttle and critical
            return (temp - self.throttle_temp) / (self.critical_temp - self.throttle_temp)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    monitor = ThermalMonitor()
    print("Monitoring Thermal Status (Press Ctrl+C to stop)...")
    try:
        while True:
            t = monitor.get_temperature()
            p = monitor.calculate_thermal_penalty()
            print(f"Temp: {t:.1f}°C | Penalty: {p:.2f}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped.")
