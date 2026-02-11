import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path
import matplotlib as mpl
from multiprocessing import Pool
import meshio
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm

# Set backend to non-interactive
mpl.use('Agg') 

# --- Color Setup ---
def get_continuous_cmap(col_list, input_hex=False, float_list=None):
    if input_hex: rgb_list = [[v/255 for v in tuple(int(c.strip("#")[i:i+2], 16) for i in (0, 2, 4))] for c in col_list]
    else: rgb_list = col_list.copy()
    if not float_list: float_list = list(np.linspace(0,1,len(rgb_list)))
    cdict = {col: [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))] 
             for num, col in enumerate(['red', 'green', 'blue'])}
    return mpl.colors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)

breaks = [-35, -11, -9, -8, -4, -2, -1]
vmin, vmax = breaks[0], breaks[-1]
cm_base = mpl.colormaps['RdYlBu_r']
col_list = [cm_base(0.15)[0:3], cm_base(0.67)[0:3], cm_base(0.8)[0:3], 
            cm_base(0.9)[0:3], mpl.colors.to_rgb('w'), (0.5, 0.5, 0.5), (0, 0, 0)]
float_list = list(mpl.colors.Normalize(vmin, vmax)(breaks))
cmap_n = get_continuous_cmap(col_list, input_hex=False, float_list=float_list)

@dataclass
class Simulation_data:
    sr: np.ndarray    
    time: np.ndarray  
    verts: np.ndarray 
    limits: list      

    @classmethod
    def load(cls, sim_dir: str, mesh_path: str, prefix: str) -> 'Simulation_data':
        base_path = Path(sim_dir)
        output_path = base_path / "output"
        
        mesh = meshio.read(mesh_path)
        triangles = mesh.cells_dict["triangle"]
        verts = mesh.points[triangles] 
        ncell = len(triangles)

        limits = [mesh.points[:,0].min(), mesh.points[:,0].max(),
                  mesh.points[:,1].min(), mesh.points[:,1].max(),
                  mesh.points[:,2].min(), mesh.points[:,2].max()]

        time_file = output_path / f"time{prefix}.dat"
        time_raw = np.loadtxt(time_file)
        time_data = (time_raw[:, 1] if time_raw.ndim > 1 else time_raw) / (365*24*60*60)

        vel_file = output_path / f"vel{prefix}.dat"
        sr_raw = np.fromfile(vel_file, dtype=np.float64).reshape(-1, ncell).T
        sr_data = np.log10(np.abs(sr_raw) + 1e-40) 

        return cls(sr=sr_data, time=time_data, verts=verts, limits=limits)

# Globals for workers
sim = None 
OUT_DIR = None

def process_frame(i):
    # Standard figure size that usually results in even pixel counts
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    norm = mpl.colors.Normalize(vmin=-12, vmax=-1)
    face_colors = cmap_n(norm(sim.sr[:, i]))
    
    poly = Poly3DCollection(sim.verts, facecolors=face_colors, edgecolors='none')
    ax.add_collection3d(poly)
    
    ax.set_xlim(sim.limits[0], sim.limits[1])
    ax.set_ylim(sim.limits[2], sim.limits[3])
    ax.set_zlim(sim.limits[4], sim.limits[5])
    
    ax.set_box_aspect([1, 1, 0.15])
    ax.view_init(elev=15, azim=-90)
    ax.set_title(f't = {sim.time[i]:.2f} yrs')
    
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap_n)
    plt.colorbar(mappable, ax=ax, shrink=0.5, label='log10(Slip Rate [m/s])')
    
    # Save frame - using a fixed filename format that matches ffmpeg's pattern
    save_path = os.path.join(OUT_DIR, f'frame_{i:06d}.png')
    plt.savefig(save_path, dpi=100)
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sim_dir', type=str)
    parser.add_argument('mesh_file', type=str)
    parser.add_argument('--sufix', type=str, default='1')
    parser.add_argument('--step', type=int, default=5)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--keep_frames', action='store_true', help='Do not delete temp_frames folder')
    args = parser.parse_args()

    OUT_DIR = "temp_frames_{}".format(args.sufix)
    if os.path.exists(OUT_DIR): shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)
    
    print(f"--> Loading Data (Prefix: {args.sufix})...")
    sim = Simulation_data.load(args.sim_dir, args.mesh_file, args.sufix)
    
    # Filter indices based on step
    indices = list(range(0, sim.sr.shape[1], args.step))
    print(f"--> Rendering {len(indices)} frames...")
    
    with Pool(processes=args.workers) as pool:
        pool.map(process_frame, indices)

    # ffmpeg stitching
    # -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" ensures dimensions are divisible by 2
    video_name = f"slip_rate_prefix_{args.sufix}.mp4"
    cmd = [
        'ffmpeg', '-y', '-framerate', '20', 
        '-pattern_type', 'glob', '-i', f'{OUT_DIR}/frame_*.png',
        '-vf', "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', video_name
    ]
    
    print(f"--> Stitching video...")
    subprocess.run(cmd)

    if not args.keep_frames:
        shutil.rmtree(OUT_DIR)
        print(f"--> Cleaned up temporary frames.")
    else:
        print(f"--> Frames preserved in {OUT_DIR}.")

    print(f"--> Done! Video: {video_name}")