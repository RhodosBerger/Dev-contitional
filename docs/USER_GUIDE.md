# User Guide

## 1. Installation

### Prerequisites
- Python 3.9+
- Intel OpenVINO (optional, for Metacognition acceleration)
- Git

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/GAMESA-KrystalStack.git
cd GAMESA-KrystalStack

# Install dependencies
pip install -r requirements.txt
```

## 2. Quick Start

### Awakening the Organism
Run the main entry point in **Demo Mode** to see the biological loops in action:
```bash
python -m src.python.main --mode demo
```
*Expected Output:*
> [Krystal-Prime] AWAKENED. The Industrial Organism is alive.
> [Krystal-Prime] Cycle 1: Heartbeat Normal. Compute Price: $100.00
> ...

### Production Mode
For real-world deployment, run in production mode:
```bash
python -m src.python.main --mode production
```

## 3. Configuration

### 3D Grid Topology
Configure the dimensions of your memory grid in `config.yaml` (coming soon):
- **Tiers**: Storage layers (L1, L2, VRAM, RAM).
- **Slots**: Temporal allocations.
- **Depth**: Compute intensity per block.

### Safety Contracts
The **Spinal Reflex** defaults to:
- Max GPU Temp: 85°C
- Max Voltage: 1.1V

*To override, you must submit a **Safety Amendment Request** (future feature).*
