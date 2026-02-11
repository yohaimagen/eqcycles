import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import shutil
from pathlib import Path
import matplotlib as mpl
from multiprocessing import Pool
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm
from typing import Dict, Any, Tuple

from eqcycles.core.data import SimulationData
from eqcycles.vis.utils import SLIP_RATE_CMAP

# --- Multiprocessing Helper ---
# To make this work with multiprocessing, the worker function should be defined at the
# top level of the module so it can be pickled and sent to other processes.
# We'll pass all the necessary data to it in a tuple.

def _process_frame_worker(args: Tuple):
    """
    A helper function for the multiprocessing pool to render a single frame.
    """
    i, sim_sr_i, sim_time_i, sim_verts, sim_limits, out_dir, cmap, norm = args
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    face_colors = cmap(norm(sim_sr_i))
    
    poly = Poly3DCollection(sim_verts, facecolors=face_colors, edgecolors='none')
    ax.add_collection3d(poly)
    
    ax.set_xlim(sim_limits[0], sim_limits[1])
    ax.set_ylim(sim_limits[2], sim_limits[3])
    ax.set_zlim(sim_limits[4], sim_limits[5])
    
    ax.set_box_aspect([1, 1, 0.15])
    ax.view_init(elev=15, azim=-90)
    ax.set_title(f't = {sim_time_i:.2f} yrs')
    
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.5)
    cbar.set_label('log10(Slip Rate [m/s])')
    
    save_path = out_dir / f'frame_{i:06d}.png'
    plt.savefig(save_path, dpi=100)
    plt.close(fig)

# --- Main Class ---

class VideoRenderer:
    """
    Handles the rendering of slip rate videos from simulation data.
    """
    def __init__(self, sim_data: SimulationData, config: Dict[str, Any] = None):
        """
        Initializes the renderer with simulation data and configuration.

        Args:
            sim_data (SimulationData): The loaded and standardized simulation data.
            config (Dict, optional): Configuration for rendering.
        """
        self.sim_data = sim_data
        self.config = {
            "vmin": -12,
            "vmax": -1,
            **(config or {})
        }
        self.cmap = SLIP_RATE_CMAP
        self.norm = mpl.colors.Normalize(vmin=self.config['vmin'], vmax=self.config['vmax'])
        
        # Set backend to a non-interactive one to prevent windows from popping up
        mpl.use('Agg')

    def render_video(self, output_path: str, step: int = 5, workers: int = 4, keep_frames: bool = False):
        """
        Orchestrates the rendering of frames and stitching them into a video.

        Args:
            output_path (str): The final path for the output MP4 video.
            step (int): The interval between frames to render (e.g., render every 5th frame).
            workers (int): The number of parallel processes to use for rendering.
            keep_frames (bool): If True, the temporary directory with PNG frames is not deleted.
        """
        output_video_path = Path(output_path)
        # Use a temporary directory next to the final video file
        temp_frames_dir = output_video_path.parent / f"temp_frames_{output_video_path.stem}"

        if temp_frames_dir.exists():
            shutil.rmtree(temp_frames_dir)
        temp_frames_dir.mkdir(parents=True)

        print(f"--> Preparing to render frames in '{temp_frames_dir}'...")

        indices = list(range(0, self.sim_data.slip_rate.shape[1], step))
        
        # Prepare arguments for each worker process
        worker_args = [
            (
                i, 
                self.sim_data.slip_rate[:, i],
                self.sim_data.time[i],
                self.sim_data.mesh_verts,
                self.sim_data.mesh_limits,
                temp_frames_dir,
                self.cmap,
                self.norm
            ) for i in indices
        ]

        print(f"--> Rendering {len(indices)} frames with {workers} workers...")
        with Pool(processes=workers) as pool:
            pool.map(_process_frame_worker, worker_args)

        print(f"--> Stitching video with ffmpeg...")
        # Use glob pattern for ffmpeg input
        frame_pattern = temp_frames_dir / 'frame_*.png'
        
        # This ffmpeg command ensures video dimensions are divisible by 2, a requirement for many codecs
        cmd = [
            'ffmpeg', '-y', '-framerate', '20',
            '-i', str(frame_pattern),
            '-vf', "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',
            str(output_video_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"--> Successfully created video: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: ffmpeg failed to create the video.")
            print(f"  Return Code: {e.returncode}")
            print(f"  STDOUT: {e.stdout}")
            print(f"  STDERR: {e.stderr}")
            # Do not clean up frames if ffmpeg failed
            keep_frames = True

        if not keep_frames:
            shutil.rmtree(temp_frames_dir)
            print(f"--> Cleaned up temporary frames.")
        else:
            print(f"--> Temporary frames preserved in {temp_frames_dir}.")
