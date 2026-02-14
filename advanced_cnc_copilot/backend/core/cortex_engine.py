import logging
from typing import Dict, Any, Generator

# Sub-Engines
try:
    from backend.cms.services.dopamine_engine import DopamineEngine
except ImportError:
    class DopamineEngine: # Mock
        pass

from backend.core.parallel_streamer import ParallelStreamer
from backend.core.evolutionary_optimizer import EvolutionaryOptimizer
from backend.core.hex_logger import HexTraceLogger

class CortexEngine:
    """
    The 'Unified Engine' that orchestrates the entire 'Open Mechanic' architecture.
    Integrates:
    - Parallel Execution (Streamer)
    - Discovery (Evolutionary Optimizer)
    - Tracing (Hex Logger)
    - Biological Logic (Dopamine Engine)
    """
    def __init__(self):
        self.logger = logging.getLogger("CortexEngine")
        
        # 1. Initialize Core Engines
        self.dopamine = DopamineEngine()
        self.streamer = ParallelStreamer(self.dopamine)
        self.optimizer = EvolutionaryOptimizer(self.dopamine) # Modified to pass dopamine engine
        self.tracer = HexTraceLogger()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1) # Added for background tasks
        
        self.logger.info("Cortex Engine Initialized. Architecture Extended.")

        for event in self.streamer.execute_profile(final_profile):
            # Pass through events
            yield event

    def run_nightly_discovery(self, profiles: list[Dict]):
        """
        Runs the optimizer on a batch of profiles to find new efficiencies overnight.
        The 'Best Functionalities' you don't know about yet.
        """
        results = []
        for p in profiles:
            best = self.optimizer.evolve_profile(p)
            results.append(best)
        return results
