import os
from PIL import Image
from pathlib import Path
import trimesh
import numpy as np
import shutil

def process_stl_files(input_dir, resolution=100):
    def normalize_stl(in_path):
        mesh = trimesh.load(in_path)

        # Skip files with too many triangles
        if mesh.faces.shape[0] > 2_000_000:
            raise ValueError(f"Mesh too complex: {mesh.faces.shape[0]} faces")

        # Scale mesh to fit into a cube of `resolution` units
        scale_factor = resolution / mesh.extents.max()
        mesh.apply_scale(scale_factor)

        # Center the mesh around origin
        mesh.apply_translation(-mesh.bounding_box.centroid)

        return mesh

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".stl"):
            print(f'🔄 Processing STL file: {filename}')
            stl_file_in = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]
            voxel_out_dir = os.path.join(input_dir, base_name)
            os.makedirs(voxel_out_dir, exist_ok=True)

            try:
                # Normalize and voxelize
                normalized_mesh = normalize_stl(stl_file_in)
                pitch = 1.0

                try:
                    # Normalize and prepare
                    normalized_mesh = normalize_stl(stl_file_in)
                    pitch = 1.0

                    dims = normalized_mesh.extents / pitch
                    vox_shape = tuple(np.ceil(dims).astype(int))
                    estimated_MB = (np.prod(vox_shape) * 1) / (1024 ** 2)

                    if estimated_MB > 500:
                        print(f"❌ Skipping {filename}: estimated voxel grid too large ({estimated_MB:.1f} MB)")
                        continue

                    # Only voxelize if it's safe
                    vox = normalized_mesh.voxelized(pitch)
                    vox_matrix = np.transpose(vox.matrix.astype(np.uint8), (2, 1, 0))
                    
                except Exception as e:
                    print(f'❌ Skipping {filename} due to error: {e}')
                    continue

                # Pad to cube
                padded = np.zeros((resolution, resolution, resolution), dtype=np.uint8)
                for i in range(3):
                    if vox_matrix.shape[i] > resolution:
                        start_crop = (vox_matrix.shape[i] - resolution) // 2
                        end_crop = start_crop + resolution
                        vox_matrix = np.take(vox_matrix, indices=range(start_crop, end_crop), axis=i)
                offset = [(resolution - s) // 2 for s in vox_matrix.shape]
                end = [offset[i] + vox_matrix.shape[i] for i in range(3)]
                padded[offset[0]:end[0], offset[1]:end[1], offset[2]:end[2]] = vox_matrix

                # Save each slice
                for z in range(resolution):
                    slice_img = (padded[:, :, z] * 255).astype(np.uint8)
                    img = Image.fromarray(slice_img, mode='L')
                    img.save(os.path.join(voxel_out_dir, f"slice_{z:03d}.png"))

                print(f'✅ Saved {resolution} slices to {voxel_out_dir}')

            except Exception as e:
                print(f'❌ Skipping {filename} due to error: {e}')
                continue

    print('🎉 All STL files processed and padded to fixed shape.')



def stack_pngs_vertically(source_dir):
    for subdir in Path(source_dir).iterdir():
        if subdir.is_dir():
            png_files = sorted(subdir.glob("*.png"))
            if not png_files:
                continue

            # Load all PNGs
            images = [Image.open(p) for p in png_files]
            widths, heights = zip(*(img.size for img in images))

            max_width = max(widths)
            total_height = sum(heights)

            stacked_img = Image.new('RGB', (max_width, total_height))
            y_offset = 0
            for img in images:
                stacked_img.paste(img, (0, y_offset))
                y_offset += img.height

            # Save stacked image in parent folder with folder name
            output_path = subdir.parent / f"{subdir.name}.png"
            stacked_img.save(output_path)
            #print(f"Saved: {output_path}")
    print('done stacking PNGs vertically')

def PreprocessSTL(CADmodel, MESHmodel, resolution=100):
    """
    Preprocess STL files by converting them to PNG images.
    This function processes both CADmodel and MESHmodel directories.
    """
    # Ensure directories exist
    Path(CADmodel).mkdir(parents=True, exist_ok=True)
    Path(MESHmodel).mkdir(parents=True, exist_ok=True)

    # Process CADmodel and MESHmodel directories
    process_stl_files(CADmodel, resolution)
    process_stl_files(MESHmodel, resolution)

    stack_pngs_vertically(CADmodel)
    stack_pngs_vertically(MESHmodel)

    # Delete subdirectories insdie both CADmodel and MESHmodel and all data inside them.
    for subdir in Path(CADmodel).iterdir():
        if subdir.is_dir():
            shutil.rmtree(subdir)
    print(f"Nuked: CADmodel subdirectories")
    for subdir in Path(MESHmodel).iterdir():
        if subdir.is_dir():
            shutil.rmtree(subdir)
    print(f"Nuked: MESHmodel subdirectories")

def PreprocessSingleFile(input_file, resolution=100, cleanup=False):
    """
    Converts a single STL file into a vertically stacked PNG image.
    Output image is saved in the same directory as the STL file,
    named the same but with a .png extension.
    """
    input_file = Path(input_file)
    assert input_file.exists() and input_file.suffix.lower() == ".stl", "Input must be an existing .stl file"

    output_dir = input_file.parent
    base_name = input_file.stem  # filename without extension

    # Create a temporary working directory
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Copy the STL file into the temp directory so process_stl_files picks it up
    temp_stl = temp_dir / input_file.name
    shutil.copy(str(input_file), str(temp_stl))

    # Process and stack
    process_stl_files(str(temp_dir), resolution)
    stack_pngs_vertically(str(temp_dir))

    # Move the final stacked image to the original folder
    for img in temp_dir.glob("*.png"):
        target_path = output_dir / f"{base_name}.png"
        shutil.move(str(img), str(target_path))
        #print(f"✅ Saved: {target_path}")

    if cleanup:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)
        #print(f"🧹 Cleaned temp: {temp_dir}")