import stltovoxel
import os
from PIL import Image
from pathlib import Path

def process_stl_files(input_dir, resolution):
    for filename in os.listdir(input_dir):
        if filename.endswith(".stl"):
            stlFileIn = os.path.join(input_dir, filename)
            voxelOutDir = os.path.join(input_dir, os.path.splitext(filename)[0])
            try:
                os.mkdir(voxelOutDir)
            except FileExistsError:
                pass
            voxelOut = os.path.join(voxelOutDir, os.path.splitext(filename)[0] + '.png')
            
            # Pass the resolution to ensure 100 slices
            stltovoxel.doExport(stlFileIn, voxelOut, resolution)
    print('done voxel convertion')

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
            print(f"Saved: {output_path}")
    print('done stacking PNGs vertically')