import os
import shutil
import argparse
import numpy as np
from PIL import Image, ImageOps
import glob
import struct

# Slice-related functions from your provided code
import slice


def clear_existing_output_folders(input_folder):
    """Delete any existing folders in the input folder."""
    for item in os.listdir(input_folder):
        item_path = os.path.join(input_folder, item)
        if os.path.isdir(item_path):
            print(f"Deleting existing folder: {item_path}")
            shutil.rmtree(item_path)


def read_stl_binary(stl_path):
    """Read binary STL file and extract triangles."""
    with open(stl_path, "rb") as file:
        # Skip the 80-byte header
        file.read(80)
        
        # Read the number of triangles
        num_triangles = struct.unpack("<I", file.read(4))[0]
        
        triangles = []
        
        for _ in range(num_triangles):
            # Each triangle has 12 floats: 3 for normal, 3 for v0, 3 for v1, 3 for v2, and 2 for attributes (ignored)
            normal = struct.unpack("<3f", file.read(12))
            v0 = struct.unpack("<3f", file.read(12))
            v1 = struct.unpack("<3f", file.read(12))
            v2 = struct.unpack("<3f", file.read(12))
            file.read(2)  # Skip the attribute byte count (2 bytes)
            
            # Append the vertices as a tuple of 3D coordinates
            triangles.append((v0, v1, v2))
        
        return triangles


def convert_mesh(mesh, resolution=100, voxel_size=None, parallel=True):
    """Convert a single mesh to voxels."""
    return convert_meshes([mesh], resolution, voxel_size, parallel)


def convert_meshes(meshes, resolution=100, voxel_size=None, parallel=True):
    """Convert multiple meshes to voxel grids."""
    mesh_min, mesh_max = slice.calculate_mesh_limits(meshes)
    scale, shift, shape = slice.calculate_scale_and_shift(mesh_min, mesh_max, resolution, voxel_size)
    vol = np.zeros(shape[::-1], dtype=np.int8)
    for mesh_ind, org_mesh in enumerate(meshes):
        slice.scale_and_shift_mesh(org_mesh, scale, shift)
        cur_vol = slice.mesh_to_plane(org_mesh, shape, parallel)
        vol[cur_vol] = mesh_ind + 1
    return vol, scale, shift


def export_pngs(voxels, output_file_path, colors):
    """Export voxel grid slices as PNG images."""
    output_file_pattern, _output_file_extension = os.path.splitext(output_file_path)

    # Delete the previous output files
    file_list = glob.glob(output_file_pattern + '_*.png')
    for file_path in file_list:
        try:
            os.remove(file_path)
        except Exception:
            print("Error while deleting file : ", file_path)

    z_size = voxels.shape[0]
    size = str(len(str(z_size + 1)))
    colors = [(0, 0, 0)] + colors
    palette = [channel for color in colors for channel in color]
    for height in range(z_size):
        print(f'Exporting PNG {height + 1}/{z_size}')
        img = Image.fromarray(voxels[height].astype('uint8'), mode='P')
        img.putpalette(palette)

        img = ImageOps.flip(img)
        path = (output_file_pattern + "_%0" + size + "d.png") % height
        img.save(path)


def export_xyz(voxels, output_file_path, scale, shift):
    """Export voxel data as XYZ coordinates."""
    voxels = voxels.astype(bool)
    output = open(output_file_path, 'w')
    for z in range(voxels.shape[0]):
        for y in range(voxels.shape[1]):
            for x in range(voxels.shape[2]):
                if voxels[z][y][x]:
                    point = (np.array([x, y, z]) / scale) + shift
                    output.write('%s %s %s\n' % tuple(point))
    output.close()


def convert_stl_file(stl_path, output_dir, resolution=100):
    """Convert a single STL file to voxel slices and export to PNG."""
    stl_name = os.path.splitext(os.path.basename(stl_path))[0]
    folder_path = os.path.join(output_dir, stl_name)
    os.makedirs(folder_path, exist_ok=True)

    # Load STL file manually
    triangles = read_stl_binary(stl_path)

    # Convert mesh to numpy array
    meshes = [np.array([np.array(tri) for tri in triangles])]
    
    # Voxelize the mesh
    vol, scale, shift = convert_meshes(meshes, resolution)

    # Save each slice as PNG
    export_pngs(vol, folder_path, [(255, 255, 255)])

    print(f"Saved PNGs to {folder_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert STL files to voxel-based PNG slices.")
    parser.add_argument("input_folder", help="Folder containing STL files")
    parser.add_argument("-o", "--output", default=None, help="Output folder (defaults to input folder)")
    parser.add_argument("-r", "--resolution", type=int, default=100, help="Voxel resolution")

    args = parser.parse_args()

    input_folder = os.path.abspath(args.input_folder)
    output_folder = os.path.abspath(args.output) if args.output else input_folder

    # Step 1: Delete all existing folders in the input folder
    clear_existing_output_folders(input_folder)

    # Step 2: Create output folder if needed
    os.makedirs(output_folder, exist_ok=True)

    # Step 3: Get STL files and convert
    stl_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".stl")]

    for stl in stl_files:
        full_path = os.path.join(input_folder, stl)
        convert_stl_file(full_path, output_folder, args.resolution)


if __name__ == "__main__":
    main()
