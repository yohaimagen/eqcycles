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

    def save(self, output_dir: str, run_id: str = "downsampled") -> None:
        """
        Saves the SimulationData to a directory in a format compatible with HBILoader.
        
        Args:
            output_dir: The directory to save the files into.
            run_id: The suffix to use for the filenames.
        """
        from pathlib import Path
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        
        # Save Mesh
        mesh_path = out / f"mesh_{run_id}.msh"
        # Explicitly use gmsh22 format as it is standard for HBI/Tandem
        # Filter cells to only include those supported or intended for simulation
        supported_cells = ["triangle", "quad", "tetra", "hexahedron"]
        filtered_cells = [
            (cell_block.type, cell_block.data) 
            for cell_block in self.mesh.cells 
            if cell_block.type in supported_cells
        ]
        if not filtered_cells:
            # Fallback to whatever is there if no standard cells found, 
            # but this helps avoid the 'line' cell error in some writers.
            filtered_cells = self.mesh.cells
            
        mesh_to_save = meshio.Mesh(
            points=self.mesh.points,
            cells=filtered_cells,
            point_data=self.mesh.point_data,
            cell_data=self.mesh.cell_data
        )
        mesh_to_save.write(mesh_path, file_format="gmsh22")
        
        # Save Coords
        np.savetxt(out / f"xyz{run_id}.dat", self.coords)
        
        # Save Time
        # HBI format usually has (idx, time_sec). We reconstruct a simple version.
        time_sec = self.time * (365 * 24 * 3600)
        time_out = np.column_stack([np.arange(len(time_sec)), time_sec])
        np.savetxt(out / f"time{run_id}.dat", time_out)
        
        # Save Catalog
        # Reconstruct HBI format if possible, but at least save the current columns
        self.catalog.to_csv(out / f"event{run_id}.dat", sep=' ', index=False, header=False)
        
        # Save Fields
        def _save_field(arr, name):
            if arr is not None:
                # Reshape back to (ntime * ncell) and save as binary
                # Original loader did: data_reshaped = data_raw.reshape(ntime, ncell).T
                # So we need to transpose back to (ntime, ncell) then flatten
                if arr.ndim == 2:
                    data_to_save = arr.T.flatten()
                else:
                    data_to_save = arr.flatten()
                data_to_save.astype(np.float64).tofile(out / f"{name}{run_id}.dat")

        _save_field(self.eq_slip, "EQslip")
        # For slip_rate, we store log10(V), but HBI expects V?
        # HBILoader does: slip_rate = np.log10(np.abs(sr_data_raw))
        # So we should save 10^slip_rate
        if self.slip_rate is not None:
            v_data = 10**self.slip_rate
            v_data.T.flatten().astype(np.float64).tofile(out / f"vel{run_id}.dat")
            
        _save_field(self.state_variable, "psi")
        _save_field(self.shear_stress, "tau")
        _save_field(self.normal_stress, "sigma")
        _save_field(self.slip, "slip")
