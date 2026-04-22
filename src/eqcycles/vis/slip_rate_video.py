import numpy as np
import matplotlib.pyplot as plt
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


def _stitch_video(frame_pattern: Path, output_path: Path, framerate: int = 20) -> bool:
    """Stitch PNG frames into an MP4 with ffmpeg. Returns True on success."""
    cmd = [
        'ffmpeg', '-y', '-framerate', str(framerate),
        '-i', str(frame_pattern),
        '-vf', "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"--> Successfully created video: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print("ERROR: ffmpeg failed to create the video.")
        print(f"  Return Code: {e.returncode}")
        print(f"  STDOUT: {e.stdout}")
        print(f"  STDERR: {e.stderr}")
        return False


# --- Matplotlib Backend ---

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
        
        # Prepare arguments for each worker process.
        # frame_num (0-based sequential) is used for filenames so ffmpeg's
        # %06d pattern can read them in order.
        worker_args = [
            (
                frame_num,
                self.sim_data.slip_rate[:, i],
                self.sim_data.time[i],
                self.sim_data.mesh_verts,
                self.sim_data.mesh_limits,
                temp_frames_dir,
                self.cmap,
                self.norm,
            )
            for frame_num, i in enumerate(indices)
        ]

        print(f"--> Rendering {len(indices)} frames with {workers} workers...")
        with Pool(processes=workers) as pool:
            pool.map(_process_frame_worker, worker_args)

        print("--> Stitching video with ffmpeg...")
        # Sequential frame names let ffmpeg use %06d pattern (no glob needed)
        frame_pattern = temp_frames_dir / 'frame_%06d.png'
        success = _stitch_video(frame_pattern, output_video_path)
        if not success:
            keep_frames = True

        if not keep_frames:
            shutil.rmtree(temp_frames_dir)
            print("--> Cleaned up temporary frames.")
        else:
            print(f"--> Temporary frames preserved in {temp_frames_dir}.")


# --- PyVista Backend ---

class PyVistaVideoRenderer:
    """
    Renders slip-rate videos using PyVista/VTK for off-screen 3-D frames.

    Frames are rendered sequentially (PyVista/VTK objects cannot be pickled
    for multiprocessing), but the plotter and mesh are built once and reused
    across all frames, making this faster than creating a new figure per frame.
    The ``workers`` parameter accepted by :meth:`render_video` is silently
    ignored — it exists only for API compatibility with :class:`VideoRenderer`.
    """

    def __init__(self, sim_data: SimulationData, config: Dict[str, Any] = None):
        """
        Args:
            sim_data: Loaded simulation data.
            config:   Optional overrides for rendering settings:

                      * ``vmin`` / ``vmax``: colour-scale bounds (default −12 / −1).
                      * ``cmap``: colormap name (default ``"plasma"``).
                      * ``z_exaggeration``: vertical stretch factor (default 5).
                      * ``window_size``: ``(width, height)`` in pixels
                        (default ``(1400, 600)``).
                      * ``framerate``: output video framerate (default 20).
        """
        self.sim_data = sim_data
        self.config: Dict[str, Any] = {
            "vmin": -12,
            "vmax": -1,
            "cmap": "plasma",
            "z_exaggeration": 5.0,
            "window_size": (1400, 600),
            "framerate": 20,
            **(config or {}),
        }

    def render_video(
        self,
        output_path: str,
        step: int = 5,
        workers: int = 4,    # kept for API compatibility; unused
        keep_frames: bool = False,
    ):
        """
        Render frames with PyVista and stitch them into an MP4 with ffmpeg.

        Args:
            output_path: Destination path for the MP4 file.
            step:        Render every *step*-th time index.
            workers:     Ignored (PyVista renders sequentially).
            keep_frames: Preserve the temporary PNG directory on exit.
        """
        try:
            import pyvista as pv
        except ImportError as exc:
            raise ImportError(
                "PyVista is required for the pyvista backend. "
                "Install it with: conda install -c conda-forge pyvista"
            ) from exc

        from eqcycles.vis.diagnostics_pyvista import _build_polydata

        output_video_path = Path(output_path)
        temp_frames_dir = (
            output_video_path.parent / f"temp_frames_{output_video_path.stem}"
        )

        if temp_frames_dir.exists():
            shutil.rmtree(temp_frames_dir)
        temp_frames_dir.mkdir(parents=True)

        print(f"--> Preparing to render frames in '{temp_frames_dir}'...")

        indices = list(range(0, self.sim_data.slip_rate.shape[1], step))
        vmin        = self.config["vmin"]
        vmax        = self.config["vmax"]
        cmap        = self.config["cmap"]
        z_exag      = self.config["z_exaggeration"]
        win_size    = self.config["window_size"]
        framerate   = self.config["framerate"]

        # ── Build mesh once ────────────────────────────────────────────────────
        mesh = _build_polydata(self.sim_data.mesh_verts)
        mesh.cell_data["slip_rate"] = self.sim_data.slip_rate[:, indices[0]]

        # ── Build plotter once (off-screen) ────────────────────────────────────
        plotter = pv.Plotter(off_screen=True, window_size=list(win_size))
        plotter.set_background("white")

        scalar_bar_args = dict(
            title="log10(Slip Rate [m/s])",
            n_labels=5,
            italic=False,
            bold=False,
            title_font_size=14,
            label_font_size=11,
            color="black",
            vertical=True,
            width=0.055,
            height=0.35,
            position_x=0.91,
            position_y=0.32,
            fmt="%.3g",
        )

        plotter.add_mesh(
            mesh,
            scalars="slip_rate",
            preference="cell",
            cmap=cmap,
            clim=[vmin, vmax],
            show_edges=False,
            lighting=True,
            smooth_shading=True,
            scalar_bar_args=scalar_bar_args,
        )

        plotter.show_bounds(
            grid="back",
            location="outer",
            ticks="outside",
            n_xlabels=4,
            n_ylabels=4,
            n_zlabels=3,
            font_size=9,
            color="gray",
            xtitle="X (m)",
            ytitle="Y (m)",
            ztitle="Z (m)",
            bold=False,
        )

        if z_exag != 1.0:
            plotter.set_scale(zscale=z_exag, reset_camera=False)

        plotter.camera_position = "xz"
        plotter.camera.elevation = 20
        plotter.reset_camera()
        plotter.camera.zoom(1.8)

        # ── Render frames ──────────────────────────────────────────────────────
        print(f"--> Rendering {len(indices)} frames with PyVista (sequential)...")
        title_actor = None
        for frame_num, i in enumerate(indices):
            time_val = float(self.sim_data.time[i])

            # Update scalar data in-place — avoids rebuilding the mesh actor
            mesh.cell_data["slip_rate"] = self.sim_data.slip_rate[:, i]

            # Update frame title
            if title_actor is not None:
                plotter.remove_actor(title_actor)
            title_actor = plotter.add_text(
                f"t = {time_val:.2f} yr",
                position=(0.4, 0.88),
                font_size=12,
                color="black",
                viewport=True,
            )

            save_path = temp_frames_dir / f"frame_{frame_num:06d}.png"
            plotter.screenshot(str(save_path))

            if frame_num % 50 == 0:
                print(f"    frame {frame_num + 1}/{len(indices)}")

        plotter.close()

        # ── Stitch with ffmpeg ─────────────────────────────────────────────────
        print("--> Stitching video with ffmpeg...")
        frame_pattern = temp_frames_dir / "frame_%06d.png"
        success = _stitch_video(frame_pattern, output_video_path, framerate=framerate)
        if not success:
            keep_frames = True

        if not keep_frames:
            shutil.rmtree(temp_frames_dir)
            print("--> Cleaned up temporary frames.")
        else:
            print(f"--> Temporary frames preserved in {temp_frames_dir}.")
