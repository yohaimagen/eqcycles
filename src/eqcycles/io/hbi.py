import numpy as np
import pandas as pd
import meshio
from pathlib import Path

from eqcycles.core.data import SimulationData
from eqcycles.io.base import BaseLoader

class HBILoader(BaseLoader):
    """
    Concrete implementation of BaseLoader for HBI binary output files.
    """
    def __init__(self, mesh_path: str):
        #validate mesh path exists
        if not Path(mesh_path).exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
        self.mesh_path = mesh_path

    def load(self, path: str, run_id: str) -> SimulationData:
        """
        Loads simulation data from HBI binary files in the specified directory.
        
        Args:
            path: Path to the directory containing the .dat files.
            run_id: The identifier suffix for the files (e.g., '1', '205').
        
        Returns:
            A standardized SimulationData object.
        
        Raises:
            FileNotFoundError: If any required file is not found.
            ValueError: If data dimensions do not match.
        """
        base_path = Path(path)
        
        # Load mesh information
        if not Path(self.mesh_path).exists():
            raise FileNotFoundError(f"Mesh file not found: {self.mesh_path}")
        mesh = meshio.read(self.mesh_path)
        
        # For visualization, we often need triangles and their vertices
        # Check if "triangle" cells exist, otherwise handle accordingly
        if "triangle" in mesh.cells_dict:
            triangles = mesh.cells_dict["triangle"]
            mesh_verts = mesh.points[triangles]
        else:
            # Handle cases where there might not be triangle cells or choose a default
            print("Warning: No 'triangle' cells found in mesh. Mesh vertices for plotting might be incomplete.")
            mesh_verts = np.array([]) # Or handle differently based on expected mesh types

        mesh_limits = [
            mesh.points[:,0].min(), mesh.points[:,0].max(),
            mesh.points[:,1].min(), mesh.points[:,1].max(),
            mesh.points[:,2].min(), mesh.points[:,2].max()
        ]
        
        # 1. Load coordinates to determine ncell
        xyz_file = base_path / f"xyz{run_id}.dat"
        if not xyz_file.exists():
            raise FileNotFoundError(f"Coordinate file not found: {xyz_file}")
            
        coords = np.loadtxt(xyz_file)
        ncell = len(coords)
    
        # 2. Load the Time vector
        time_file = base_path / f"time{run_id}.dat"
        if not time_file.exists():
            raise FileNotFoundError(f"Time file not found: {time_file}")
        
        time_raw = np.loadtxt(time_file)
        # Handle both 1D and 2D time files
        time_data = (time_raw[:, 1] if time_raw.ndim > 1 else time_raw) / (365*24*60*60)
        
        ntime = len(time_data)

        # 3. Helper function to load, reshape, and transpose fields
        def _load_and_reshape_field(filename: str) -> np.ndarray:
            fpath = base_path / filename
            if not fpath.exists():
                raise FileNotFoundError(f"File not found: {fpath}")
            
            data_raw = np.fromfile(fpath, dtype=np.float64)
            
            # Reshape (-1, ncell) makes it (N_time, N_cell)
            # .T makes it (N_cell, N_time)
            data_reshaped = data_raw.reshape(ntime, ncell).T
            
            if data_reshaped.shape[1] != ntime:
                raise ValueError(
                    f"Dimension mismatch for {filename}: expected {ntime} timesteps, "
                    f"got {data_reshaped.shape[1]}. Raw data length: {len(data_raw)}, "
                    f"ncell: {ncell}, ntime: {ntime}"
                )
                
            return data_reshaped

        # 4. Load all fields
        # Note: slip_rate is log10(abs(vel))
        sr_data_raw = _load_and_reshape_field(f"vel{run_id}.dat")
        slip_rate = np.log10(np.abs(sr_data_raw)) # Add small epsilon to avoid log(0)
        state_variable = _load_and_reshape_field(f"psi{run_id}.dat")
        shear_stress = _load_and_reshape_field(f"tau{run_id}.dat")
        normal_stress = _load_and_reshape_field(f"sigma{run_id}.dat")
        slip = _load_and_reshape_field(f"slip{run_id}.dat")
        
        # Load EQ slip data
        eq_slip_file = base_path / f"EQslip{run_id}.dat"
        if not eq_slip_file.exists():
            raise FileNotFoundError(f"EQslip file not found: {eq_slip_file}")
        
        eq_slip = np.fromfile(eq_slip_file, dtype=np.float64)
        num_events = len(eq_slip) // ncell
        if num_events * ncell != len(eq_slip):
            raise ValueError(f"EQslip data length ({len(eq_slip)}) not divisible by ncell ({ncell}).")
        
        eq_slip = eq_slip.reshape(num_events, ncell).T
        
        # Load catalog data
        catalog_file = base_path / f"event{run_id}.dat"
        if not catalog_file.exists():
            raise FileNotFoundError(f"Event catalog file not found: {catalog_file}")
            
        catalog = pd.read_csv(catalog_file, sep='\s+', header=None)
        # Attempt to dynamically set columns based on expected number + any extra
        expected_cols = ['Event_ID', 'Step', 'Time_sec', 'Mw', 'Hypo_Node']
        if catalog.shape[1] < len(expected_cols):
             raise ValueError(f"Catalog file {catalog_file} has fewer columns than expected.")
        
        catalog.columns = expected_cols + [f'Col_{i}' for i in range(len(expected_cols), catalog.shape[1])]
        catalog['Time_year'] = catalog['Time_sec'] / (365*24*60*60)
        
        return SimulationData(
            slip_rate=slip_rate,
            state_variable=state_variable,
            shear_stress=shear_stress,
            normal_stress=normal_stress,
            slip=slip,
            time=time_data,
            coords=coords,
            mesh=mesh,
            mesh_verts=mesh_verts,
            mesh_limits=mesh_limits,
            eq_slip=eq_slip,
            catalog=catalog
        )
