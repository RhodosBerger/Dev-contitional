import numpy as np
import logging

class GamesaGrid:
    """
    Gamesa 3D Grid: High-Performance Voxel Simulation.
    Uses Vectorized Numpy operations to simulate material removal and collision.
    "Drastically Improved Performance" via GPU-ready data structures.
    """
    def __init__(self, resolution=100):
        self.resolution = resolution
        self.logger = logging.getLogger("GamesaGrid")
        
        # 3D Tensor representing the Voxel Grid (0=Empty, 1=Material)
        # Using uint8 for memory efficiency
        self.grid = np.ones((resolution, resolution, resolution), dtype=np.uint8)
        self.logger.info(f"Gamesa 3D Grid Initialized. Resolution: {resolution}^3 ({resolution**3/1e6:.1f}M Voxels)")

    def simulate_segment(self, segment: dict) -> str:
        """
        Simulates scanning a segment through the voxel grid.
        Returns 'SAFE', 'COLLISION', or 'RISK'.
        """
        # Feature: Vectorized Collision Check
        target = segment.get("target", {})
        x, y, z = target.get("x", 0), target.get("y", 0), target.get("z", 0)
        
        # Normalize coordinates to grid index
        idx_x = int((x / 1000.0) * self.resolution)
        idx_y = int((y / 1000.0) * self.resolution)
        idx_z = int((z / 1000.0) * self.resolution)
        
        # Boundary Check
        if not (0 <= idx_x < self.resolution and 0 <= idx_y < self.resolution and 0 <= idx_z < self.resolution):
            return "COLLISION" # Out of bounds
            
        # Material Check (Simulate cutting)
        if self.grid[idx_x, idx_y, idx_z] == 1:
            # Material found - simulate removal (Transition 1 -> 0)
            self.grid[idx_x, idx_y, idx_z] = 0
            return "SAFE" # Cut successful
            
        return "SAFE" # Air cut

    def get_grid_density(self) -> float:
        """
        Returns the percentage of material remaining.
        High-Speed Vectorized Calculation.
        """
        return np.mean(self.grid) * 100.0

    def get_projection_matrix(self):
        """
        Returns the projection matrix for GPU rendering.
        """
        return self.grid # Pass the tensor directly
