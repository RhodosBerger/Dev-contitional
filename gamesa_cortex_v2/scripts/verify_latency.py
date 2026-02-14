import time
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from gamesa_cortex_v2.src.core.npu_coordinator import NPUCoordinator

def verify_latency():
    print("Initializing NPU Coordinator...")
    npu = NPUCoordinator()
    
    # 1. Warmup
    print("Warming up JIT/Cache...")
    for _ in range(10):
        npu.dispatch_task(lambda: 2+2, "WARMUP", 100)
        
    # 2. Latency Test
    print("\nStarting Latency Test (1000 iterations)...")
    start_time = time.perf_counter_ns()
    
    iterations = 1000
    for i in range(iterations):
        # Simulate a fast "NPU Inference" task
        future = npu.dispatch_task(lambda: 2**10, "AI_INFERENCE", 50)
        if future:
            future.result() # Wait for completion to measure full round-trip
            
    end_time = time.perf_counter_ns()
    
    total_time_ms = (end_time - start_time) / 1_000_000.0
    avg_latency_ms = total_time_ms / iterations
    effective_fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0
    
    print(f"\nRESULTS:")
    print(f"Total Time: {total_time_ms:.2f} ms")
    print(f"Avg Latency: {avg_latency_ms:.4f} ms per task")
    print(f"Effective Throughput: {effective_fps:.2f} FPS")
    
    if avg_latency_ms < 1.0:
        print("[SUCCESS] Sub-millisecond latency achieved!")
    else:
        print("[WARNING] Latency > 1ms. Optimization needed.")

if __name__ == "__main__":
    verify_latency()
