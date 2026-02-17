# ASCII Compositor & AI Adaptation - Quick Reference Guide

**Companion to:** ASCII_COMPOSITOR_AI_ADAPTATION_STUDY.md  
**Framework:** Krystal-Stack Platform  
**Last Updated:** February 17, 2026

---

## 1. Core Concepts at a Glance

### Active Optic Compositor (AOC)
🎯 **Purpose:** Transform visual scenes into system health metrics  
📊 **Output:** Entropy metrics (spatial, temporal, frequency)  
🔄 **Feedback:** Drives adaptive resource allocation

### Visual Entropy
```
Low Entropy (0.0-0.3)   → System Healthy/Stable
Medium Entropy (0.3-0.7) → System Moderate Load
High Entropy (0.7-1.0)   → System Degrading/Failing
```

### Visual Backpressure
```
High Entropy → Throttle Resources → Reduce Entropy → Restore Performance
```

---

## 2. Quick Implementation Checklist

### Step 1: Import AOC
```python
from backend.cms.active_optic_compositor import ActiveOpticCompositor, EntropyMetrics
```

### Step 2: Initialize
```python
compositor = ActiveOpticCompositor()
```

### Step 3: Generate Visual Data
```python
# Convert your telemetry to visual field
# Example: Servo errors as 3D array
visual_data = np.array([...])  # Shape: (height, width, channels)
```

### Step 4: Calculate Entropy
```python
metrics = compositor.calculate_entropy(visual_data)
print(f"Total Entropy: {metrics.total_entropy:.2f}")
print(f"Coherence: {metrics.coherence:.2f}")
```

### Step 5: Make Decision
```python
if compositor.should_throttle(metrics):
    # Reduce system load
    throttle_background_tasks()
```

---

## 3. ASCII Scene Generation

### Workflow
```
Telemetry → Neuro-Geometric Model → ASCII Scene → Entropy Validation → Display
```

### Example Code
```python
def generate_system_status_ascii(cpu, memory, gpu_temp):
    """Generate ASCII scene from telemetry"""
    scene = f"""
┌───────────────────────────────┐
│   SYSTEM STATUS               │
├───────────────────────────────┤
│ CPU:    {render_bar(cpu)}    │
│ Memory: {render_bar(memory)} │
│ GPU:    {render_bar(gpu_temp)}│
└───────────────────────────────┘
"""
    
    # Validate entropy
    visual_field = ascii_to_visual_array(scene)
    entropy = compositor.calculate_entropy(visual_field)
    
    if entropy.total_entropy > 0.7:
        # Simplify scene
        scene = simplify_ascii(scene)
    
    return scene

def render_bar(value, max_val=100, width=20):
    """Render progress bar"""
    filled = int((value / max_val) * width)
    bar = '█' * filled + '░' * (width - filled)
    return f"{bar} {value:>3.0f}%"
```

---

## 4. Entropy Thresholds

### Recommended Thresholds by Application

| Application              | Throttle Threshold | Critical Threshold | Action              |
|--------------------------|--------------------|--------------------|---------------------|
| **CNC Machining**        | 0.65               | 0.85               | Emergency Stop      |
| **Robotic Assembly**     | 0.70               | 0.90               | Slow Mode           |
| **AI Inference Server**  | 0.75               | 0.95               | Load Balancing      |
| **SCADA Monitoring**     | 0.60               | 0.80               | Alert Operator      |
| **General Purpose**      | 0.70               | 0.90               | Adaptive Throttle   |

---

## 5. Integration Points

### With Gamesa Cortex V2

```python
# In Rust Safety-Critical Planner
entropy_cost = entropy_metrics.total_entropy * RISK_WEIGHT
path_cost = base_cost + entropy_cost

# In Economic Governor
budget_multiplier = entropy_metrics.coherence  # 0.0-1.0
allocated_budget = base_budget * budget_multiplier

# In vGPU Manager
if entropy_metrics.total_entropy > 0.7:
    isolate_workload_to_vgpu_slice(workload_id, priority='HIGH')
```

### With FANUC RISE v3.0

```python
# Shadow Council Validation
def shadow_council_review(proposed_action):
    # Creator Agent proposes
    scene = creator_agent.generate_ascii_scene(proposed_action)
    
    # Auditor Agent validates entropy
    entropy = compositor.calculate_entropy(ascii_to_visual(scene))
    if entropy.total_entropy > SAFETY_THRESHOLD:
        return "REJECTED: High Entropy"
    
    # Accountant Agent checks resources
    if resource_cost(scene) > budget:
        return "REJECTED: Over Budget"
    
    return "APPROVED"
```

---

## 6. Common Patterns

### Pattern 1: Real-Time Monitoring
```python
while True:
    telemetry = read_sensor_data()
    ascii_scene = generate_ascii_scene(telemetry)
    
    print("\033[2J\033[H", end="")  # Clear terminal
    print(ascii_scene)
    
    entropy = get_scene_entropy(ascii_scene)
    if entropy.total_entropy > 0.7:
        alert_operator(entropy)
    
    time.sleep(0.1)  # 10 Hz update
```

### Pattern 2: Predictive Maintenance
```python
entropy_history = []

for frame in data_stream:
    metrics = compositor.calculate_entropy(frame)
    entropy_history.append(metrics.total_entropy)
    
    # Detect entropy trend
    if len(entropy_history) > 100:
        trend = np.polyfit(range(100), entropy_history[-100:], deg=1)[0]
        
        if trend > 0.01:  # Increasing entropy
            time_to_failure = estimate_failure_time(trend)
            schedule_maintenance(time_to_failure)
```

### Pattern 3: Adaptive Rendering
```python
def adaptive_ascii_renderer(data, target_entropy=0.5):
    """Render with automatic quality adjustment"""
    
    # Start with high detail
    detail_level = 'HIGH'
    
    while True:
        scene = render_ascii(data, detail=detail_level)
        entropy = get_scene_entropy(scene)
        
        if entropy.total_entropy < target_entropy:
            break
        
        # Reduce detail if entropy too high
        detail_level = reduce_detail(detail_level)
    
    return scene
```

---

## 7. Debugging Tips

### Issue: Entropy always high
**Causes:**
- Poor visual field normalization
- Incompatible data types (int vs float)
- Frame history not accumulating

**Solutions:**
```python
# Normalize visual data to [0, 1]
visual_data = (data - data.min()) / (data.max() - data.min() + 1e-10)

# Ensure correct dtype
visual_data = visual_data.astype(np.float32)

# Check frame history
print(f"History size: {len(compositor.frame_history)}")
```

### Issue: Temporal entropy is zero
**Cause:** Only one frame in history

**Solution:**
```python
# Feed multiple frames before checking temporal entropy
for i in range(10):
    compositor.calculate_entropy(frames[i])

# Now temporal entropy is valid
metrics = compositor.calculate_entropy(current_frame)
```

### Issue: Entropy doesn't correlate with system state
**Causes:**
- Wrong mapping from data to visual field
- Incorrect entropy component weights

**Solutions:**
```python
# Experiment with different mappings
visual_field = create_visual_field(data, mapping='heatmap')  # vs 'bars', 'graph'

# Adjust entropy weights
total_entropy = (spatial * 0.5 +  # Increase spatial weight
                temporal * 0.2 + 
                frequency * 0.3)
```

---

## 8. Performance Optimization

### Baseline Performance
- **Entropy Calculation:** ~2 ms per frame (10^4 elements)
- **ASCII Generation:** ~3 ms (integer-only neural network)
- **Total Latency:** ~5 ms (suitable for 100 Hz control loops)

### Optimization Strategies

#### 1. Reduce Visual Field Size
```python
# Downsample before entropy calculation
visual_field_small = downsample(visual_field, factor=2)
metrics = compositor.calculate_entropy(visual_field_small)
```

#### 2. Skip Frequency Entropy
```python
# If frequency domain not needed
spatial = compositor._calculate_spatial_entropy(frame)
temporal = compositor._calculate_temporal_entropy()
total = (spatial * 0.6 + temporal * 0.4)  # Skip FFT
```

#### 3. Batch Processing
```python
# Process multiple frames together (GPU acceleration)
metrics_batch = [compositor.calculate_entropy(f) for f in frames]
```

#### 4. Approximate Entropy
```python
# Use sampling for large visual fields
sample_indices = np.random.choice(field.size, size=1000)
sampled_field = field.flatten()[sample_indices].reshape((-1, 1, 1))
metrics = compositor.calculate_entropy(sampled_field)
```

---

## 9. Safety Guidelines

### For Safety-Critical Systems

✅ **DO:**
- Use deterministic ASCII rendering (no GPU dependencies)
- Log entropy values to safety audit trail
- Set conservative entropy thresholds
- Implement dual-channel entropy calculation (redundancy)
- Test with worst-case scenarios (max entropy inputs)

❌ **DON'T:**
- Rely solely on visual entropy for safety decisions
- Skip validation of AI-generated ASCII scenes
- Use floating-point operations in safety kernels
- Exceed certified latency budgets

### Example Safety Architecture
```python
# Dual-channel entropy calculation
entropy_channel_a = compositor_a.calculate_entropy(visual_data)
entropy_channel_b = compositor_b.calculate_entropy(visual_data)

# Cross-check
if abs(entropy_channel_a.total_entropy - 
       entropy_channel_b.total_entropy) > 0.05:
    raise SafetyException("Entropy calculation divergence")

# Conservative decision
max_entropy = max(entropy_channel_a.total_entropy,
                 entropy_channel_b.total_entropy)

if max_entropy > SAFETY_LIMIT:
    trigger_safe_state()
```

---

## 10. Resources

### Code Locations
- **AOC Implementation:** `advanced_cnc_copilot/backend/cms/active_optic_compositor.py`
- **ASCII Rendering:** `legacy_docs/HEXADECIMAL_SYSTEM_README.md`
- **Example Usage:** `active_optic_compositor.py` (root)

### Documentation
- **Full Study:** `docs/ASCII_COMPOSITOR_AI_ADAPTATION_STUDY.md`
- **Visual Theory:** `legacy_docs/VISUAL_OPTIC_SYSTEM_STUDY.md`
- **FANUC Integration:** `advanced_cnc_copilot/backend/cms/theories/FANUC_WAVE_ARCHITECTURE.md`

### Key Equations

**Total Entropy:**
```
E_total = 0.4·E_spatial + 0.3·E_temporal + 0.3·E_frequency
```

**Coherence:**
```
Coherence = 1.0 - E_total
```

**Throttle Factor:**
```
F_throttle = 1.0 - ((E_total - T_threshold) / (1.0 - T_threshold)) · (1.0 - F_min)
```

**Visual Backpressure:**
```
IF E_total > T_threshold THEN
    Resource_Allocation = Base_Allocation · (1.0 - E_total)
END
```

---

## 11. Frequently Asked Questions

### Q: Can I use RGB images instead of ASCII?
**A:** Yes, the AOC works with any visual field (RGB images, depth maps, heatmaps). ASCII is recommended for interpretability and edge deployment.

### Q: What if my system doesn't have visual output?
**A:** You can map any time-series data to a visual field. Example: sensor array → 2D heatmap → entropy calculation.

### Q: How do I choose entropy thresholds?
**A:** Start with 0.7 for throttling, 0.9 for critical alerts. Calibrate using labeled "healthy" vs "failing" data.

### Q: Can entropy calculation run on microcontrollers?
**A:** Yes, but use optimized implementations:
- Skip FFT (frequency entropy)
- Use fixed-point arithmetic
- Reduce frame history size

### Q: How does this relate to neural network training?
**A:** Entropy metrics can serve as loss function components, guiding the network to generate low-entropy (interpretable) outputs.

---

**End of Quick Reference**

*For detailed explanations, see the full study: ASCII_COMPOSITOR_AI_ADAPTATION_STUDY.md*
