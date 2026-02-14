import time
import sys
import logging
from backend.cms.services.dopamine_engine import DopamineEngine
from backend.repositories.telemetry_repository import TelemetryRepository

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("KrystalStackBenchmark")

def run_bandwidth_test():
    logger.info("Starting Krystal Stack PC Benchmark...")
    
    # 1. Initialize Stack
    repo = TelemetryRepository()
    engine = DopamineEngine(repo)
    
    # Warup
    logger.info("Warming up GPU Stack...")
    dummy_metrics = {"spindle_load": 50, "temperature": 40, "vibration_x": 0.5}
    for _ in range(10):
        engine.detect_phantom_trauma(1, dummy_metrics)
        
    # 2. Bandwidth Test (Inference Loop)
    iterations = 1000
    start_time = time.time()
    
    for i in range(iterations):
        # Simulate high-frequency "Analog" sensor data
        metrics = {
            "spindle_load": 50 + (i % 20),
            "temperature": 40 + (i % 10),
            "vibration_x": 0.5 + (i % 100) / 100.0
        }
        engine.detect_phantom_trauma(1, metrics)
        
    end_time = time.time()
    duration = end_time - start_time
    fps = iterations / duration
    
    logger.info("="*40)
    logger.info(f"BENCHMARK RESULTS (PC LINUX STACK)")
    logger.info("="*40)
    logger.info(f"Total Inferences: {iterations}")
    logger.info(f"Duration:         {duration:.4f} seconds")
    logger.info(f"Throughput (FPS): {fps:.2f} inferences/sec")
    logger.info("="*40)
    
    if fps > 500:
        logger.info("RESULT: EXCELLENT (High Bandwidth Achieved)")
    else:
        logger.info("RESULT: NOMINAL (Standard Performance)")

if __name__ == "__main__":
    try:
        run_bandwidth_test()
    except KeyboardInterrupt:
        sys.exit(0)
