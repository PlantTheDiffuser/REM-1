import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import ConvertSTLtoVoxel as conv

resolution = 100  # Number of slices/images per file
convert = False  # Set to True if you need to convert STL files to PNG
if convert:
    # Get current script directory
    current_dir = Path(__file__).resolve().parent

    CADmodel = str(current_dir / 'PreflightCheckTrainingData/CADmodel')
    MESHmodel = str(current_dir / 'PreflightCheckTrainingData/MESHmodel')

    # Process CADmodel and MESHmodel directories
    conv.process_stl_files(CADmodel, resolution)
    conv.process_stl_files(MESHmodel, resolution)

