from eqcycles.io.hbi import HBILoader
import numpy as np
from pathlib import Path
import os

# Create dummy output directory and files for testing purposes
# In a real scenario, these files would already exist.
dummy_output_dir = "./output_test"
Path(dummy_output_dir).mkdir(parents=True, exist_ok=True)

# Create dummy files that HBILoader expects
# These files will be minimal and just enough to allow the loader to run without FileNotFoundError
# The actual content doesn't matter for this structural test, just their existence and basic format.

# xyz.dat: ncell lines, 3 columns
with open(Path(dummy_output_dir) / "xyz7.dat", "w") as f:
    f.write("0.0 0.0 0.0\n")
    f.write("1.0 1.0 1.0\n")
    f.write("2.0 2.0 2.0\n")
dummy_ncell = 3

# time.dat: ntime lines, 2 columns (or 1D)
with open(Path(dummy_output_dir) / "time7.dat", "w") as f:
    f.write("0 0.0\n")
    f.write("1 31536000.0\n") # 1 year in seconds
    f.write("2 63072000.0\n") # 2 years in seconds
dummy_ntime = 3

# vel.dat, psi.dat, tau.dat, sigma.dat: binary float64, ntime * ncell values
# Using numpy to create dummy binary files
dummy_data_shape = (dummy_ntime, dummy_ncell)

np.zeros(dummy_data_shape, dtype=np.float64).tofile(Path(dummy_output_dir) / "vel7.dat")
np.zeros(dummy_data_shape, dtype=np.float64).tofile(Path(dummy_output_dir) / "psi7.dat")
np.zeros(dummy_data_shape, dtype=np.float64).tofile(Path(dummy_output_dir) / "tau7.dat")
np.zeros(dummy_data_shape, dtype=np.float64).tofile(Path(dummy_output_dir) / "sigma7.dat")

# EQslip.dat: binary float64, num_events * ncell values
dummy_num_events = 2
np.zeros((dummy_num_events, dummy_ncell), dtype=np.float64).tofile(Path(dummy_output_dir) / "EQslip7.dat")

# event.dat: catalog data, sep='\s+', header=None
with open(Path(dummy_output_dir) / "event7.dat", "w") as f:
    f.write("1 0 0.0 7.0 10\n")
    f.write("2 1 31536000.0 6.5 20\n")


# Dummy mesh file using meshio to create a valid, minimal .msh
dummy_mesh_path = "./NAF.msh"
import meshio
points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
cells = [("triangle", [[0, 1, 2], [1, 3, 2]])]
mesh = meshio.Mesh(points, cells)
mesh.write(dummy_mesh_path)


print("Attempting to load data using HBILoader...")
try:
    loader = HBILoader(mesh_path=dummy_mesh_path)
    sim_data = loader.load(path=dummy_output_dir, run_id='7')
    print(f"Successfully loaded data.")
    print(f"sim_data.slip_rate.shape: {sim_data.slip_rate.shape}")
    print(f"sim_data.time.shape: {sim_data.time.shape}")
    print(f"sim_data.catalog.head():\n{sim_data.catalog.head()}")
    
    # Test subset_time
    subset = sim_data.subset_time(0.5, 1.5)
    print(f"\nSubset data (time 0.5-1.5 years):")
    print(f"subset.slip_rate.shape: {subset.slip_rate.shape}")
    print(f"subset.time: {subset.time}")
    print(f"subset.catalog:\n{subset.catalog}")

except FileNotFoundError as e:
    print(f"Test failed due to FileNotFoundError: {e}")
    print("Please ensure all dummy data files (xyz7.dat, time7.dat, vel7.dat, etc.) and the mesh file (NAF.msh) exist in the specified paths.")
except ValueError as e:
    print(f"Test failed due to ValueError: {e}")
except Exception as e:
    print(f"An unexpected error occurred during loading: {e}")
finally:
    # Clean up dummy files and directory
    print(f"\nCleaning up dummy files and directory: {dummy_output_dir}, {dummy_mesh_path}")
    if os.path.exists(dummy_output_dir):
        import shutil
        shutil.rmtree(dummy_output_dir)
    if os.path.exists(dummy_mesh_path):
        os.remove(dummy_mesh_path)
    print("Cleanup complete.")
