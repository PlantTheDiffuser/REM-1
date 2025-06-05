import torch
import os
import numpy
import stltovoxel
from pathlib import Path
import ConvertSTLtoVoxel as conv

resolution = 100  # Number of slices/images per file

# Get current script directory
current_dir = Path(__file__).resolve().parent

CADmodel = str(current_dir / 'PreflightCheckTrainingData/CADmodel')
MESHmodel = str(current_dir / 'PreflightCheckTrainingData/MESHmodel')

# Process CADmodel and MESHmodel directories
conv.process_stl_files(CADmodel, resolution)
conv.process_stl_files(MESHmodel, resolution)

print('done')