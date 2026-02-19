from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
import meshio # Assuming meshio objects will be stored directly

@dataclass
class SimulationData:
    """
    A standardized container for simulation output data.
    """
    # Standardized arrays
    time: np.ndarray         # time from time*.dat (in years)
    coords: np.ndarray       # xyz coordinates from xyz*.dat

    # Mesh information
    mesh: meshio.Mesh        # The full meshio Mesh object
    mesh_verts: np.ndarray   # Vertices from mesh.cells_dict["triangle"] for plotting
    mesh_limits: list        # [xmin, xmax, ymin, ymax, zmin, zmax] of mesh points

    # Earthquake related data
    eq_slip: np.ndarray      # EQslip*.dat data
    catalog: pd.DataFrame    # event*.dat data as a Pandas DataFrame
    
    # Optional heavy fields
    slip_rate: Optional[np.ndarray] = None    # log10(abs(vel)) from vel*.dat
    state_variable: Optional[np.ndarray] = None # psi from psi*.dat
    shear_stress: Optional[np.ndarray] = None # tau from tau*.dat
    normal_stress: Optional[np.ndarray] = None # sigma from sigma*.dat (added from plot_eq_sequance.py)
    slip: Optional[np.ndarray] = None        # slip from slip*.dat (added for diagnostics)

    def subset_time(self, t_start: float, t_end: float) -> 'SimulationData':
        """
        Returns a new SimulationData object sliced by time.
        Time values are expected to be in years.
        """
        if t_start is None:
            t_start = self.time.min()
        if t_end is None:
            t_end = self.time.max()

        time_mask = (self.time >= t_start) & (self.time <= t_end)
        
        # Ensure that time_mask is not empty
        if not np.any(time_mask):
            raise ValueError("No data found within the specified time range.")

        # Determine the number of timesteps after masking
        num_timesteps_masked = np.sum(time_mask)

        # Helper to safely subset data arrays that have time as the last dimension
        def _subset_if_time_dependent(arr):
            if arr is None:
                return None
            # Check if the last dimension matches the original number of timesteps
            # before applying the mask. This handles arrays that might not have a
            # time dimension (e.g., coords, which is (N_cells, 3))
            if arr.shape[-1] == self.time.shape[0]:
                return arr[..., time_mask]
            return arr

        # Apply subsetting to all time-dependent arrays
        return SimulationData(
            time=self.time[time_mask],
            coords=self.coords, # Coords are not time-dependent
            mesh=self.mesh, # Mesh object is not time-dependent
            mesh_verts=self.mesh_verts, # Mesh vertices are not time-dependent
            mesh_limits=self.mesh_limits, # Mesh limits are not time-dependent
            eq_slip=_subset_if_time_dependent(self.eq_slip),
            catalog=self.catalog[(self.catalog['Time_year'] >= t_start) & (self.catalog['Time_year'] <= t_end)].reset_index(drop=True),
            slip_rate=_subset_if_time_dependent(self.slip_rate),
            state_variable=_subset_if_time_dependent(self.state_variable),
            shear_stress=_subset_if_time_dependent(self.shear_stress),
            normal_stress=_subset_if_time_dependent(self.normal_stress),
            slip=_subset_if_time_dependent(self.slip)
        )