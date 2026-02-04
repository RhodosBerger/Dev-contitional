# API Reference

## `src.python.organism`

### `IndustrialOrganism`
The central lifecycle manager.

#### `__init__(name: str)`
Initializes the biological subsystems (Economy, Immune System, Brain).

#### `awaken()`
Starts the main heartbeat loop.

---

## `src.python.economics`

### `CrossForexMarket`
The internal resource exchange.

#### `submit_order(order: MarketOrder) -> bool`
Submits a bid/ask for a resource. Returns `True` if queued successfully.

#### `get_market_price(asset_type: str) -> float`
Returns the current dynamic price of a resource based on supply/demand.

---

## `src.python.guardian`

### `SpinalReflex` (Layer 1 Safety)
#### `check_reflex(telemetry: dict) -> bool`
Runs deterministic safety checks. Returns `False` if an immediate shutdown is required.

### `CorticalMonitor` (Layer 2 Safety)
#### `analyze_patterns(history: List[dict])`
Analyzes long-term usage trends to optimize safety policies.
