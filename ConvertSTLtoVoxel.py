import stltovoxel
import os


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
    print('done')