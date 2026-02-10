"""
Fanuc Mock Server 🖥️
Simulates a Fanuc CNC machine for testing the API without hardware.
"""
from fastapi import FastAPI
import uvicorn
import random

app = FastAPI(title="Fanuc CNC Simulator")

state = {
    "status": "IDLE",
    "pos": {"X": 0.0, "Y": 0.0, "Z": 0.0},
    "alarms": []
}

@app.get("/focas2/status")
def get_status():
    # Simulate some movement
    if state["status"] == "RUNNING":
        state["pos"]["X"] += random.uniform(-1, 1)
        state["pos"]["Y"] += random.uniform(-1, 1)
    return state

@app.post("/focas2/command")
def send_command(cmd: str):
    if cmd == "START":
        state["status"] = "RUNNING"
    elif cmd == "STOP":
        state["status"] = "IDLE"
    return {"result": "OK"}

def start_server():
    uvicorn.run(app, host="0.0.0.0", port=8193)

if __name__ == "__main__":
    start_server()
