import numpy as np

import sys
from pathlib import Path

# Add the folder containing this script to the search path
script_dir = Path(__file__).parent.absolute()
sys.path.append(str(script_dir))
import construction as cnst

def save_points_as_ply(points_tuple, filename):
    # Unpack and flatten our 1000x1000 grids into long lists of x, y, z
    x, y, z = [np.array(p).flatten() for p in points_tuple]
    
    with open(filename, 'w') as f:
        # The PLY Header (tells the viewer what's inside)
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(x)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        
        # Write every coordinate pair
        for i in range(len(x)):
            f.write(f"{x[i]} {y[i]} {z[i]}\n")

# --- EXECUTION ---
sphere_data, cylinder_data, cone_data = cnst.construct()

save_points_as_ply(sphere_data, "gauss_sphere.ply")
save_points_as_ply(cylinder_data, "gauss_cylinder.ply")
save_points_as_ply(cone_data, "gauss_cone.ply")

print("Files generated! Go open them in 3D Viewer.")