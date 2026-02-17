#!/usr/bin/env python3
"""
ASCII Entropy Demonstration
Companion script for ASCII_COMPOSITOR_AI_ADAPTATION_STUDY.md

Demonstrates:
1. Entropy calculation from ASCII scenes
2. AI-driven ASCII scene generation
3. Visual backpressure mechanism
4. Adaptive system response
"""

import numpy as np
import time
from typing import List, Tuple
from dataclasses import dataclass

# Try to import AOC from the actual codebase
try:
    import sys
    sys.path.insert(0, '/home/dusan/Documents/GitHub/Dev-contitional/advanced_cnc_copilot')
    from backend.cms.active_optic_compositor import ActiveOpticCompositor, EntropyMetrics
    AOC_AVAILABLE = True
except ImportError:
    print("⚠️  Active Optic Compositor not available, using simplified version")
    AOC_AVAILABLE = False
    
    # Simplified implementation for demonstration
    @dataclass
    class EntropyMetrics:
        spatial_entropy: float
        temporal_entropy: float
        frequency_entropy: float
        total_entropy: float
        coherence: float
    
    class ActiveOpticCompositor:
        def __init__(self):
            self.frame_history = []
            self.max_history = 100
        
        def calculate_entropy(self, visual_data: np.ndarray) -> EntropyMetrics:
            self.frame_history.append(visual_data)
            if len(self.frame_history) > self.max_history:
                self.frame_history.pop(0)
            
            spatial = min(1.0, np.var(visual_data))
            
            if len(self.frame_history) >= 2:
                diffs = [np.mean(np.abs(self.frame_history[i] - self.frame_history[i-1])) 
                        for i in range(1, len(self.frame_history))]
                temporal = min(1.0, np.var(diffs) / 0.1)
            else:
                temporal = 0.0
            
            frequency = 0.5  # Simplified
            
            total = (spatial * 0.4 + temporal * 0.3 + frequency * 0.3)
            coherence = 1.0 - total
            
            return EntropyMetrics(spatial, temporal, frequency, total, coherence)
        
        def should_throttle(self, metrics: EntropyMetrics, threshold: float = 0.7) -> bool:
            return metrics.total_entropy > threshold


def ascii_to_visual_array(ascii_scene: str, width: int = 80, height: int = 24) -> np.ndarray:
    """Convert ASCII scene to visual array for entropy calculation"""
    lines = ascii_scene.strip().split('\n')
    
    # Create visual field
    visual = np.zeros((height, width, 1), dtype=np.float32)
    
    for i, line in enumerate(lines[:height]):
        for j, char in enumerate(line[:width]):
            # Map character to intensity
            # More complex characters = higher intensity
            intensity = ord(char) / 255.0 if char else 0.0
            visual[i, j, 0] = intensity
    
    return visual


def generate_stable_ascii_scene() -> str:
    """Generate low-entropy ASCII scene (healthy system)"""
    return """
┌────────────────────────────────────────────────────────┐
│                 SYSTEM STATUS (STABLE)                 │
├────────────────────────────────────────────────────────┤
│ CPU:      ████████████░░░░░░░░  60%                   │
│ Memory:   ██████████░░░░░░░░░░  50%                   │
│ GPU:      ████████████████░░░░  80%                   │
│ Disk I/O: ██████░░░░░░░░░░░░░░  30%                   │
├────────────────────────────────────────────────────────┤
│ Active Tasks: 12                                       │
│ Errors: 0                                              │
│ Uptime: 24h 15m                                        │
└────────────────────────────────────────────────────────┘
"""


def generate_chaotic_ascii_scene() -> str:
    """Generate high-entropy ASCII scene (failing system)"""
    return """
┌────────────────────────────────────────────────────────┐
│            SYSTEM STATUS (⚠️  CRITICAL)                │
├────────────────────────────────────────────────────────┤
│ CPU:      █▓▒░▓█░▒▓░█▓░▒█▓░▒█  98% ⚠️                 │
│ Memory:   ▓█▒░▓█░▒▓█░▒▓█░▒▓█  97% ⚠️                 │
│ GPU:      █▒░█▓░▒█░▓▒█░▓▒█░▓  99% ⚠️                 │
│ Disk I/O: ░▓█▒░▓█▒░▓█▒░▓█▒░▓  95% ⚠️                 │
├────────────────────────────────────────────────────────┤
│ Active Tasks: 247                                      │
│ Errors: 38 ⚠️⚠️⚠️                                      │
│ Uptime: 0h 3m (CRASHED)                                │
└────────────────────────────────────────────────────────┘
"""


def generate_adaptive_scene(load: float) -> str:
    """Generate ASCII scene that adapts to system load (AI-driven)"""
    bar_width = 20
    filled = int(load * bar_width)
    bar = '█' * filled + '░' * (bar_width - filled)
    
    # Adaptive complexity based on load
    if load < 0.5:
        status = "OPTIMAL"
        icon = "✓"
    elif load < 0.8:
        status = "MODERATE"
        icon = "!"
    else:
        status = "CRITICAL"
        icon = "⚠"
    
    return f"""
┌──────────────────────────────────────┐
│ {icon} System Load: {status:^10} {icon}      │
├──────────────────────────────────────┤
│ Load: {bar} {load*100:>3.0f}%   │
└──────────────────────────────────────┘
"""


def simulate_system_operation(compositor: ActiveOpticCompositor, duration: int = 10):
    """Simulate system operation with entropy-driven adaptation"""
    print("\n" + "="*60)
    print("SIMULATION: Entropy-Driven System Adaptation")
    print("="*60)
    
    system_load = 0.3  # Start stable
    load_velocity = 0.05  # Load change rate
    
    for t in range(duration):
        print(f"\n--- Time Step {t+1}/{duration} ---")
        
        # Generate scene based on current load
        if system_load < 0.4:
            scene = generate_stable_ascii_scene()
            scene_type = "STABLE"
        elif system_load < 0.75:
            scene = generate_adaptive_scene(system_load)
            scene_type = "ADAPTIVE"
        else:
            scene = generate_chaotic_ascii_scene()
            scene_type = "CRITICAL"
        
        # Convert to visual field and calculate entropy
        visual = ascii_to_visual_array(scene)
        metrics = compositor.calculate_entropy(visual)
        
        print(f"Scene Type: {scene_type}")
        print(f"System Load: {system_load:.2f}")
        print(f"Entropy Metrics:")
        print(f"  - Spatial:  {metrics.spatial_entropy:.3f}")
        print(f"  - Temporal: {metrics.temporal_entropy:.3f}")
        print(f"  - Total:    {metrics.total_entropy:.3f}")
        print(f"  - Coherence: {metrics.coherence:.3f}")
        
        # Adaptive response based on entropy
        if compositor.should_throttle(metrics):
            print("🔻 BACKPRESSURE TRIGGERED: Reducing system load...")
            system_load = max(0.2, system_load - 0.15)  # Reduce load
            load_velocity = -0.05  # Reverse direction
        else:
            print("✓ System healthy, maintaining course")
            system_load = min(0.95, system_load + load_velocity)
        
        # Simulate load variation
        load_velocity += np.random.uniform(-0.02, 0.02)
        load_velocity = np.clip(load_velocity, -0.1, 0.1)
        
        time.sleep(0.5)  # Slow down for readability


def benchmark_entropy_calculation():
    """Benchmark entropy calculation performance"""
    print("\n" + "="*60)
    print("BENCHMARK: Entropy Calculation Performance")
    print("="*60)
    
    sizes = [
        (10, 10, 1),    # Small
        (80, 24, 1),    # Standard terminal
        (160, 48, 1),   # Large terminal
        (320, 96, 1),   # Very large
    ]
    
    for size in sizes:
        # Create fresh compositor for each size to avoid shape mismatch
        compositor = ActiveOpticCompositor()
        
        # Generate random visual field
        visual_data = np.random.rand(*size).astype(np.float32)
        
        # Warm-up
        compositor.calculate_entropy(visual_data)
        
        # Benchmark
        iterations = 100
        start = time.time()
        for _ in range(iterations):
            compositor.calculate_entropy(visual_data)
        end = time.time()
        
        avg_time = (end - start) / iterations * 1000  # Convert to ms
        elements = size[0] * size[1] * size[2]
        
        print(f"\nSize: {size} ({elements:,} elements)")
        print(f"Average time: {avg_time:.3f} ms")
        print(f"Throughput: {elements/avg_time:.0f} elements/ms")


def demonstrate_ai_generation():
    """Demonstrate AI-driven ASCII scene generation with entropy optimization"""
    print("\n" + "="*60)
    print("DEMONSTRATION: AI-Driven ASCII Scene Generation")
    print("="*60)
    
    compositor = ActiveOpticCompositor()
    
    # Simulate telemetry data
    telemetry_states = [
        {"cpu": 0.25, "memory": 0.30, "gpu": 0.40, "errors": 0},
        {"cpu": 0.65, "memory": 0.70, "gpu": 0.75, "errors": 2},
        {"cpu": 0.95, "memory": 0.92, "gpu": 0.98, "errors": 15},
    ]
    
    for i, telemetry in enumerate(telemetry_states):
        print(f"\n--- Telemetry State {i+1} ---")
        print(f"CPU: {telemetry['cpu']*100:.0f}%, "
              f"Memory: {telemetry['memory']*100:.0f}%, "
              f"GPU: {telemetry['gpu']*100:.0f}%, "
              f"Errors: {telemetry['errors']}")
        
        # Generate scene
        avg_load = (telemetry['cpu'] + telemetry['memory'] + telemetry['gpu']) / 3
        scene = generate_adaptive_scene(avg_load)
        
        print("\nGenerated ASCII Scene:")
        print(scene)
        
        # Calculate entropy
        visual = ascii_to_visual_array(scene)
        metrics = compositor.calculate_entropy(visual)
        
        print(f"Scene Entropy: {metrics.total_entropy:.3f}")
        print(f"Scene Coherence: {metrics.coherence:.3f}")
        
        # Validate
        if metrics.total_entropy > 0.7:
            print("⚠️  Scene entropy too high, refinement recommended")
        else:
            print("✓ Scene entropy acceptable")


def compare_scenes():
    """Compare stable vs chaotic scenes side-by-side"""
    print("\n" + "="*60)
    print("COMPARISON: Stable vs Chaotic ASCII Scenes")
    print("="*60)
    
    compositor = ActiveOpticCompositor()
    
    print("\n--- STABLE SCENE ---")
    stable = generate_stable_ascii_scene()
    print(stable)
    
    stable_visual = ascii_to_visual_array(stable)
    stable_metrics = compositor.calculate_entropy(stable_visual)
    
    print(f"\nStable Metrics:")
    print(f"  Total Entropy: {stable_metrics.total_entropy:.3f}")
    print(f"  Coherence:     {stable_metrics.coherence:.3f}")
    
    # Reset history for fair comparison
    compositor.frame_history = []
    
    print("\n--- CHAOTIC SCENE ---")
    chaotic = generate_chaotic_ascii_scene()
    print(chaotic)
    
    chaotic_visual = ascii_to_visual_array(chaotic)
    chaotic_metrics = compositor.calculate_entropy(chaotic_visual)
    
    print(f"\nChaotic Metrics:")
    print(f"  Total Entropy: {chaotic_metrics.total_entropy:.3f}")
    print(f"  Coherence:     {chaotic_metrics.coherence:.3f}")
    
    print("\n--- ANALYSIS ---")
    print(f"Entropy Increase: {chaotic_metrics.total_entropy - stable_metrics.total_entropy:.3f}")
    print(f"Coherence Drop:   {stable_metrics.coherence - chaotic_metrics.coherence:.3f}")


def main():
    """Run all demonstrations"""
    print("\n" + "="*60)
    print(" ASCII COMPOSITOR & AI ADAPTATION DEMONSTRATION")
    print(" Companion to: ASCII_COMPOSITOR_AI_ADAPTATION_STUDY.md")
    print("="*60)
    
    if not AOC_AVAILABLE:
        print("\n⚠️  Using simplified AOC implementation for demonstration")
    
    # Menu
    print("\nAvailable Demonstrations:")
    print("1. Compare Stable vs Chaotic Scenes")
    print("2. AI-Driven Scene Generation")
    print("3. System Operation Simulation (Entropy-Driven Adaptation)")
    print("4. Entropy Calculation Benchmark")
    print("5. Run All")
    
    choice = input("\nSelect demonstration (1-5): ").strip().replace('.', '')
    
    try:
        if choice == '1':
            compare_scenes()
        elif choice == '2':
            demonstrate_ai_generation()
        elif choice == '3':
            compositor = ActiveOpticCompositor()
            simulate_system_operation(compositor, duration=10)
        elif choice == '4':
            benchmark_entropy_calculation()
        elif choice == '5':
            compare_scenes()
            demonstrate_ai_generation()
            compositor = ActiveOpticCompositor()
            simulate_system_operation(compositor, duration=5)
            benchmark_entropy_calculation()
        else:
            print(f"Invalid choice '{choice}', running comparison demo...")
            compare_scenes()
    except Exception as e:
        print(f"\n❌ ERROR during demonstration: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Demonstration complete!")
    print("For more details, see: docs/ASCII_COMPOSITOR_AI_ADAPTATION_STUDY.md")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
