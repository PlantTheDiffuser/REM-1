import torch
import os
import numpy
import stltovoxel
from pathlib import Path

resolution = 100

# Get current script directory
current_dir = Path(__file__).resolve().parent

CADmodel = str(current_dir / 'PreflightCheckTrainingData/CADmodel')
MESHmodel = str(current_dir / 'PreflightCheckTrainingData/MESHmodel')

# Example: list files
for filename in os.listdir(CADmodel):
    if filename.endswith(".stl"):
        stlFileIn = os.path.join(CADmodel, filename)
        voxelOut = CADmodel + '/' + os.path.splitext(filename)[0]
        try:
            os.mkdir(voxelOut)
        except FileExistsError:
            pass
        voxelOut = voxelOut + '/' + os.path.splitext(filename)[0] + '.png'
        stltovoxel.doExport(stlFileIn, voxelOut, resolution)

for filename in os.listdir(MESHmodel):
    if filename.endswith(".stl"):
        stlFileIn = os.path.join(MESHmodel, filename)
        voxelOut = MESHmodel + '/' + os.path.splitext(filename)[0]
        try:
            os.mkdir(voxelOut)
        except FileExistsError:
            pass
        voxelOut = voxelOut + '/' + os.path.splitext(filename)[0] + '.png'
        stltovoxel.doExport(stlFileIn, voxelOut, resolution)

print('done')