# Component Reference

## 1. Hexadecimal Trading System
A precise, low-level implementation of the Economic Engine.

### Hex Commodity Types
| Hex Range | Type | Description |
| :--- | :--- | :--- |
| **0x00-0x1F** | `HEX_COMPUTE` | raw CPU cycles |
| **0x20-0x3F** | `HEX_MEMORY` | VRAM or RAM blocks |
| **0x60-0x7F** | `HEX_GPU` | Shader units |
| **0x80-0x9F** | `HEX_NEURAL` | NPU/Tensor cores |

### Usage
```python
# Create a bid for 100 units of Compute
bid = HexCommodity(type=HEX_COMPUTE, amount=100, price=0.50)
market.submit(bid)
```

## 2. ASCII Rendering Engine
A visualization tool for terminal-based monitoring.
- **Hex Digit Rendering**: Displays resources as stylized ASCII hex codes.
- **Market State**: Visualizes the order book as a depth chart.

## 3. OpenVINO Integration
Accelerates "Metacognition" (AI analysis) on Intel hardware.
- **Function**: Compiles PyTorch models to Intermediate Representation (IR) for rapid inference on CPUs/iGPUs.
- **Use Case**: Real-time anomaly detection in telemetry streams.

## 4. Composition Generator
A creative engine that "composes" optimal resource states.
- **Pattern Matching**: Detects if the current workload is "Gaming", "Training", or "Idle".
- **Fibonacci Scaling**: Allocates resources using natural growth ratios (1, 1, 2, 3, 5...) for organic scaling.
