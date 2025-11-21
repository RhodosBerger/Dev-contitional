# GAMESA/KrystalStack - Cross-Forex Resource Market Engine

> A revolutionary real-time hardware optimization engine architected as a high-frequency trading market. System resources—CPU/GPU cycles, thermal headroom, memory bandwidth—are treated as tradable assets managed by autonomous AI agents.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     GUARDIAN LAYER (Python)                         │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────────┐ │
│  │ Metacognitive│  │  Experience  │  │    Signal Scheduler         │ │
│  │  Interface   │  │    Store     │  │  (Domain-Ranked Priority)   │ │
│  └──────┬──────┘  └──────┬───────┘  └─────────────┬───────────────┘ │
│         │                │                        │                 │
│  ┌──────▼────────────────▼────────────────────────▼───────────────┐ │
│  │                    EFFECT CHECKER                              │ │
│  │         (Capability Validation & Audit Trail)                  │ │
│  └──────────────────────────┬─────────────────────────────────────┘ │
│                             │                                       │
│  ┌──────────────────────────▼─────────────────────────────────────┐ │
│  │                  CONTRACT VALIDATOR                            │ │
│  │      (Pre/Post/Invariant Checks, Self-Healing)                 │ │
│  └──────────────────────────┬─────────────────────────────────────┘ │
└─────────────────────────────┼───────────────────────────────────────┘
                              │ IPC / Shared Schemas
┌─────────────────────────────▼───────────────────────────────────────┐
│                   DETERMINISTIC STREAM (Rust)                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ Economic Engine │  │   ActionGate    │  │   Rule Evaluator    │  │
│  │ (Cross-Forex    │  │ (Safety Guard-  │  │ (MicroInference     │  │
│  │  Market Core)   │  │  rails/Limits)  │  │  Rules/Shadow)      │  │
│  └────────┬────────┘  └────────┬────────┘  └──────────┬──────────┘  │
│           │                    │                      │             │
│  ┌────────▼────────────────────▼──────────────────────▼──────────┐  │
│  │                      ALLOCATOR                                │  │
│  │    (Resource Pools: CPU/GPU/Memory/Thermal/Power Budgets)     │  │
│  └───────────────────────────┬───────────────────────────────────┘  │
│                              │                                      │
│  ┌───────────────────────────▼───────────────────────────────────┐  │
│  │                    RUNTIME                                    │  │
│  │  (Variable Fetch, Feature Engine, Expression Evaluation)      │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Concepts

### Cross-Forex Resource Market

Resources are treated as tradable assets with economic profiles:

| Resource | Unit | Strategy | Use Case |
|----------|------|----------|----------|
| CPU Cores | cores | FirstFit | Task pinning |
| CPU Time | μs | BestFit | Scheduling |
| GPU Compute | % | Buddy | VRAM allocation |
| GPU Memory | MB | Buddy | Power-of-two blocks |
| Thermal Headroom | °C | Priority | Safety margin |
| Power Budget | mW | Weighted | TDP management |

### Autonomous Agents

Each hardware component has an agent that trades resources:

```python
from src.python import (
    create_default_allocator, AllocationRequest, ResourceType, Priority,
    SignalScheduler, telemetry_signal, SignalKind
)

# Create the "stock exchange"
allocator = create_default_allocator()

# CPU Agent requests resources
cpu_request = AllocationRequest(
    id="cpu-agent-001",
    resource_type=ResourceType.CPU_CORES,
    amount=4,
    priority=Priority.HIGH,
)
allocation = allocator.allocate(cpu_request)

# Signal scheduler processes market signals
scheduler = SignalScheduler()
scheduler.enqueue(telemetry_signal(SignalKind.CPU_BOTTLENECK, strength=0.8))
signal = scheduler.dequeue()  # Highest priority wins
```

### Guardian/Hex Central Bank

The Guardian regulates trades and ensures system stability:

```python
from src.python import (
    create_guardian_checker, create_guardian_validator,
    Effect, SAFETY_CONTRACT
)

# Effect-based capability control
checker = create_guardian_checker()
if checker.can_perform("cpu_agent", Effect.CPU_CONTROL):
    # Agent has permission
    pass

# Contract-based validation
validator = create_guardian_validator()
result = validator.check_invariants("safety_check", {
    "temp_cpu": 72, "temp_gpu": 68
})
```

## Module Reference

### Python (`src/python/`)

| Module | Description |
|--------|-------------|
| `metacognitive.py` | Self-reflecting LLM analysis |
| `experience_store.py` | S,A,R tuple storage |
| `policy_generator.py` | Rule generation |
| `feature_engine.py` | α/β/θ scaling, trig transforms |
| `runtime.py` | Live variable fetching |
| `allocation.py` | Resource pool management |
| `effects.py` | Capability tokens |
| `contracts.py` | Pre/post/invariant validation |
| `signals.py` | Domain-ranked scheduling |

### Rust (`src/rust/`)

| Module | Description |
|--------|-------------|
| `types.rs` | Core schemas |
| `economic_engine.rs` | Cost/benefit scoring |
| `action_gate.rs` | Safety guardrails |
| `rule_evaluator.rs` | Condition evaluation |
| `feature_engine.rs` | Expression parser |
| `runtime.rs` | Telemetry integration |
| `allocation.rs` | Multi-strategy allocation |
| `effects.rs` | Effect declaration |
| `contracts.rs` | Contract validation |
| `signals.rs` | Signal priority queue |

## Quick Start

```bash
# Rust
cargo build && cargo test

# Python
pip install -r requirements.txt
python examples/cross_forex_demo.py
```

## Telemetry → Decision → Learning Loop

```
Telemetry ──► Signal Scheduler ──► Domain Ranking ──► Economic Engine
    │                                                        │
    │                                                        ▼
    │                                              ActionGate Filter
    │                                                        │
    └───────────────── ExperienceStore ◄──────────── Execute Action
                            │
                            ▼
                   Metacognitive Analysis
```

## Samples

See `samples/` for configurations:
- `policy_proposal.json` - LLM proposal
- `micro_inference_rule.json` - Declarative rule
- `allocation_schema.json` - Pool configs
- `scale_presets.json` - α/β/θ presets

## License

MIT
