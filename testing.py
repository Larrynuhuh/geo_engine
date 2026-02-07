import numpy as np
import construction as cnst

def save_points_as_ply(points_tuple, filename):
    # 1. Convert JAX to Numpy and FLATTEN (1000x1000 -> 1,000,000)
    x = np.array(points_tuple[0]).flatten() # index 0 is x
    y = np.array(points_tuple[1]).flatten() # index 1 is y
    z = np.array(points_tuple[2]).flatten() # index 2 is z
    
    # 2. SUB-SAMPLE: Take every 10th point (1,000,000 -> 10,000)
    # This makes the file much lighter so the 3D Viewer doesn't crash
    x, y, z = x[::100], y[::100], z[::100]
    
    # 3. Write the file
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(x)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        
        for i in range(len(x)):
            f.write(f"{x[i]} {y[i]} {z[i]}\n")

# Execute
sphere_data, cylinder_data, cone_data = cnst.construct()
save_points_as_ply(sphere_data, "gauss_sphere.ply")
print("New lightweight file generated!")
