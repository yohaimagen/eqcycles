#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

from eqcycles.io.hbi import HBILoader
from eqcycles.vis.slip_rate_video import VideoRenderer

def main():
    parser = argparse.ArgumentParser(description="Generate a slip rate video from HBI simulation data.")
    
    # Positional arguments
    parser.add_argument("sim_dir", type=str, help="Directory containing HBI simulation files.")
    parser.add_argument("run_id", type=str, help="Run ID suffix for the simulation files (e.g., '1').")
    parser.add_argument("mesh_path", type=str, help="Path to the mesh file (e.g., .msh or .h5).")
    parser.add_argument("output", type=str, nargs='?', default=None, help="Path to save the output MP4 video (default: slip_rate_{run_id}.mp4).")
    
    # Optional arguments
    parser.add_argument("--step", type=int, default=5, help="Frame step interval (default: 5).")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers for rendering (default: 4).")
    parser.add_argument("--vmin", type=float, default=-12.0, help="Minimum log10 slip rate for colorbar (default: -12).")
    parser.add_argument("--vmax", type=float, default=-1.0, help="Maximum log10 slip rate for colorbar (default: -1).")
    parser.add_argument("--keep_frames", action="store_true", help="Keep the temporary directory of PNG frames.")
    
    args = parser.parse_args()

    sim_dir = Path(args.sim_dir)
    if not sim_dir.exists():
        print(f"Error: Simulation directory '{args.sim_dir}' does not exist.")
        sys.exit(1)
        
    mesh_path = Path(args.mesh_path)
    if not mesh_path.exists():
        print(f"Error: Mesh file '{args.mesh_path}' does not exist.")
        sys.exit(1)

    if args.output is None:
        output_path = Path(f"slip_rate_{args.run_id}.mp4")
    else:
        output_path = Path(args.output)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".mp4")

    print(f"--> Loading simulation data from {sim_dir} (Run ID: {args.run_id})...")
    try:
        loader = HBILoader(mesh_path=str(mesh_path))
        # We only need slip_rate for the video
        sim_data = loader.load(
            path=str(sim_dir), 
            run_id=args.run_id, 
            fields_to_load=['slip_rate']
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    print("--> Data loaded successfully. Initializing renderer...")
    config = {
        "vmin": args.vmin,
        "vmax": args.vmax
    }
    
    try:
        renderer = VideoRenderer(sim_data, config=config)
        renderer.render_video(
            output_path=str(output_path),
            step=args.step,
            workers=args.workers,
            keep_frames=args.keep_frames
        )
    except Exception as e:
        print(f"Error during video rendering: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
