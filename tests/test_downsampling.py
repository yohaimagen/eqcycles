import numpy as np
import pytest
import meshio
from eqcycles.core.data import SimulationData
from eqcycles.analysis.downsampling import downsample_simulation, calculate_triangle_areas, get_triangle_centroids
import pandas as pd

def create_simple_mesh(nx, ny, x_len=10.0, y_len=10.0):
    """Creates a simple rectangular mesh of triangles."""
    x = np.linspace(0, x_len, nx)
    y = np.linspace(0, y_len, ny)
    xv, yv = np.meshgrid(x, y)
    points = np.column_stack([xv.ravel(), yv.ravel(), np.zeros_like(xv.ravel())])
    
    triangles = []
    for i in range(ny - 1):
        for j in range(nx - 1):
            # Two triangles per quad
            p0 = i * nx + j
            p1 = i * nx + (j + 1)
            p2 = (i + 1) * nx + j
            p3 = (i + 1) * nx + (j + 1)
            triangles.append([p0, p1, p2])
            triangles.append([p1, p3, p2])
            
    cells = [("triangle", np.array(triangles))]
    return meshio.Mesh(points, cells)

@pytest.fixture
def synthetic_data():
    # High-res mesh: 11x11 points -> 10x10 quads -> 200 triangles
    mesh_high = create_simple_mesh(11, 11)
    ncell = len(mesh_high.cells_dict["triangle"])
    coords = get_triangle_centroids(mesh_high)
    
    # Simple time dependent data
    ntime = 5
    time = np.linspace(0, 1, ntime)
    
    # Field: constant 1.0 everywhere
    slip = np.ones((ncell, ntime))
    
    # Catalog
    catalog = pd.DataFrame({
        'Event_ID': [1],
        'Step': [100],
        'Time_sec': [3600],
        'Mw': [6.0],
        'Hypo_Node': [0], # First cell
        'Time_year': [3600 / (365*24*3600)]
    })
    
    # eq_slip: (ncell, num_events)
    eq_slip = np.ones((ncell, 1))
    
    data = SimulationData(
        time=time,
        coords=coords,
        mesh=mesh_high,
        mesh_verts=mesh_high.points[mesh_high.cells_dict["triangle"]],
        mesh_limits=[0, 10, 0, 10, 0, 0],
        eq_slip=eq_slip,
        catalog=catalog,
        slip=slip,
        slip_rate=np.ones((ncell, ntime)) * -9.0, # log10(V)
        shear_stress=np.ones((ncell, ntime)) * 10.0
    )
    return data

def test_area_calculation():
    # 1x1 quad -> 2 triangles. Total area should be 1.0.
    mesh = create_simple_mesh(2, 2, x_len=1.0, y_len=1.0)
    areas = calculate_triangle_areas(mesh)
    assert len(areas) == 2
    assert np.allclose(np.sum(areas), 1.0)

def test_downsampling_conservation(synthetic_data):
    # Low-res mesh: 6x6 points -> 5x5 quads -> 50 triangles
    mesh_low = create_simple_mesh(6, 6)
    
    downsampled = downsample_simulation(synthetic_data, mesh_low)
    
    # 1. Shape check
    n_target = len(mesh_low.cells_dict["triangle"])
    assert downsampled.slip.shape == (n_target, 5)
    assert downsampled.coords.shape == (n_target, 3)
    
    # 2. Conservation of constant field
    # Since original field was constant 1.0, downsampled should be ~1.0
    assert np.allclose(downsampled.slip, 1.0)
    assert np.allclose(downsampled.eq_slip, 1.0)
    
    # 3. Area conservation check
    source_areas = calculate_triangle_areas(synthetic_data.mesh)
    target_areas = calculate_triangle_areas(mesh_low)
    assert np.allclose(np.sum(source_areas), np.sum(target_areas))

def test_catalog_mapping(synthetic_data):
    mesh_low = create_simple_mesh(6, 6)
    downsampled = downsample_simulation(synthetic_data, mesh_low)
    
    # Hypo_Node was 0 in source (at [0.33, 0.33, 0])
    # Target mesh first centroid should be at [0.66, 0.66, 0] or similar
    # It should still be a valid index
    assert 0 <= downsampled.catalog['Hypo_Node'].iloc[0] < len(downsampled.coords)

def test_z_filtering():
    # Create a mesh that crosses Z=-18
    points = np.array([
        [0, 0, -10], [1, 0, -10], [0, 1, -10],
        [0, 0, -20], [1, 0, -20], [0, 1, -20]
    ])
    cells = [("triangle", np.array([[0, 1, 2], [3, 4, 5]]))]
    mesh_high = meshio.Mesh(points, cells)
    
    coords = get_triangle_centroids(mesh_high)
    ncell = 2
    data = SimulationData(
        time=np.array([0]),
        coords=coords,
        mesh=mesh_high,
        mesh_verts=points[cells[0][1]],
        mesh_limits=[0, 1, 0, 1, -20, -10],
        eq_slip=np.array([[1.0], [2.0]]), # Cell 0 (Z=-10) is 1.0, Cell 1 (Z=-20) is 2.0
        catalog=pd.DataFrame({'Hypo_Node': [0]})
    )
    
    # Target mesh with one cell at Z=-10
    target_points = np.array([[0, 0, -10.1], [1, 0, -10.1], [0, 1, -10.1]])
    target_cells = [("triangle", np.array([[0, 1, 2]]))]
    target_mesh = meshio.Mesh(target_points, target_cells)
    
    # Downsample with Z limit at -15
    downsampled = downsample_simulation(data, target_mesh, z_limit=-15)
    
    # Target cell is at -10.1 (above -15). 
    # It should only take from source cell 0 (-10), not cell 1 (-20).
    # Value should be 1.0, not 1.5 (average).
    assert np.allclose(downsampled.eq_slip[0, 0], 1.0)

def test_saving(synthetic_data, tmp_path):
    mesh_low = create_simple_mesh(6, 6)
    output_dir = tmp_path / "downsampled_output"
    
    downsampled = downsample_simulation(
        synthetic_data, 
        mesh_low, 
        output_dir=str(output_dir),
        run_id="test_run"
    )
    
    # Check if files exist
    assert (output_dir / "xyztest_run.dat").exists()
    assert (output_dir / "timetest_run.dat").exists()
    assert (output_dir / "eventtest_run.dat").exists()
    assert (output_dir / "EQsliptest_run.dat").exists()
    assert (output_dir / "veltest_run.dat").exists()
    
    # Verify content of xyz (should match downsampled coords)
    xyz_saved = np.loadtxt(output_dir / "xyztest_run.dat")
    assert np.allclose(xyz_saved, downsampled.coords)
