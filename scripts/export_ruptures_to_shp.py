#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

from eqcycles.io.hbi import HBILoader
from eqcycles.analysis.geometry import project_to_fault_trace
from eqcycles.analysis.rupture import export_ruptures_to_geodataframe

def main():
    parser = argparse.ArgumentParser(description="Export simulation ruptures to a polyline shapefile.")
    
    # Positional arguments
    parser.add_argument("sim_dir", type=str, help="Directory containing HBI simulation files.")
    parser.add_argument("run_id", type=str, help="Run ID suffix for the simulation files.")
    parser.add_argument("mesh_path", type=str, help="Path to the mesh file.")
    parser.add_argument("shapefile_path", type=str, help="Path to the reference fault trace shapefile.")
    parser.add_argument("output_shp", type=str, nargs='?', default=None, help="Output shapefile path (default: ruptures_{run_id}.shp).")
    
    # Optional arguments
    parser.add_argument("--out_dir", type=str, default=".", help="Directory to save the output files (default: current directory).")
    parser.add_argument("--min_mw", type=float, default=7.0, help="Minimum magnitude to export (default: 7.0).")
    parser.add_argument("--threshold", type=float, default=0.05, help="Slip threshold (m) to define rupture (default: 0.05).")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output.")
    
    args = parser.parse_args()

    sim_dir = Path(args.sim_dir)
    if not sim_dir.exists():
        print(f"Error: Simulation directory '{args.sim_dir}' does not exist.")
        sys.exit(1)
        
    mesh_path = Path(args.mesh_path)
    if not mesh_path.exists():
        print(f"Error: Mesh file '{args.mesh_path}' does not exist.")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.output_shp is None:
        output_shp = out_dir / f"ruptures_{args.run_id}.shp"
    else:
        output_shp = Path(args.output_shp)
        if not output_shp.is_absolute():
            output_shp = out_dir / output_shp
            
        if not output_shp.suffix:
            output_shp = output_shp.with_suffix(".shp")

    print(f"--> Loading simulation data from {sim_dir} (Run ID: {args.run_id})...")
    try:
        loader = HBILoader(mesh_path=str(mesh_path))
        # We need coords, catalog and eq_slip
        sim_data = loader.load(
            path=str(sim_dir), 
            run_id=args.run_id
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    print(f"--> Projecting mesh to fault trace: {args.shapefile_path}...")
    try:
        mesh_along_strike = project_to_fault_trace(sim_data.coords, args.shapefile_path)
    except Exception as e:
        print(f"Error during projection: {e}")
        sys.exit(1)

    print(f"--> Transforming ruptures to polylines...")
    try:
        gdf = export_ruptures_to_geodataframe(
            sim_data=sim_data,
            mesh_along_strike=mesh_along_strike,
            shapefile_path=args.shapefile_path,
            slip_threshold=args.threshold,
            min_mw=args.min_mw,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"Error during rupture transformation: {e}")
        sys.exit(1)

    if gdf.empty:
        print("--> No ruptures found to export. Exiting.")
        sys.exit(0)

    print(f"--> Saving to shapefile: {output_shp}...")
    try:
        gdf.to_file(output_shp)
        print(f"--> Successfully exported {len(gdf)} ruptures.")
    except Exception as e:
        print(f"Error saving shapefile: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
