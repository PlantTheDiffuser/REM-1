import os
import subprocess

# Get current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the data folder path
CADmodel = os.path.join(current_dir, "PreflightCheckTrainingData/CADmodel")
MESHmodel = os.path.join(current_dir, "PreflightCheckTrainingData/MESHmodel")

# Example: list files
for filename in os.listdir(CADmodel):
    if filename.endswith(".stl"):
        print(filename)
for filename in os.listdir(MESHmodel):
    if filename.endswith(".stl"):
        print(filename)


# Example: run the conversion with args
convertscript = os.path.join(current_dir, "ConvertSTLtoVoxel.py")
subprocess.run([
    "python3", convertscript,
    CADmodel,
    "-r", "100"
])

print("done")