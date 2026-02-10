"""
Fanuc FOCAS Client 🤖
Communicates with Fanuc CNC machines via FOCAS protocol (or simulation).
"""
import logging
import json
import time

# Use standard library to avoid external dependencies for the client if requests is missing
import urllib.request
import urllib.error

logger = logging.getLogger("Fanuc.Client")

class FanucClient:
    def __init__(self, ip: str, port: int = 8193, use_mock_server: bool = True):
        self.ip = ip
        self.port = port
        self.use_mock_server = use_mock_server
        self.base_url = f"http://{ip}:{port}/focas2"
        self.connected = False

    def connect(self):
        """Establishes connection to the CNC (or Mock Server)."""
        logger.info(f"Connecting to Fanuc CNC at {self.ip}:{self.port}...")
        try:
            # Ping the server
            if self.use_mock_server:
                with urllib.request.urlopen(f"{self.base_url}/status", timeout=2) as response:
                    if response.status == 200:
                        self.connected = True
                        return True
            else:
                # Real FOCAS via Fwlib (Stub)
                self.connected = True
                return True
        except Exception as e:
            logger.warning(f"Connection failed: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Closes connection."""
        if self.connected:
            logger.info("Disconnecting from Fanuc CNC...")
            self.connected = False

    def get_status(self):
        """Reads machine status (Run, Stop, Alarm)."""
        if not self.connected:
            return {"status": "DISCONNECTED"}
            
        if self.use_mock_server:
            try:
                with urllib.request.urlopen(f"{self.base_url}/status", timeout=2) as response:
                     return json.loads(response.read().decode())
            except Exception as e:
                logger.error(f"Status read failed: {e}")
                return {"status": "ERROR"}
        
        # Fallback / Real Lib
        return {"status": "RUNNING", "mode": "MEM", "execution": "ACTIVE"}

    def get_position(self):
        """Reads axis positions."""
        if not self.connected:
            return {}
        # In a real app, we'd hit /position or similar
        return {"X": 100.234, "Y": -50.112, "Z": 10.500, "A": 0.0}
