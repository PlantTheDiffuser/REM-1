from pathlib import Path
import shutil
import ConvertSTLtoVoxel as conv
import PreflightCheckTrainer as PreflightCheck
import ReverseEngineeringModel as REM

# ---------- Configuration ----------
FileToRun = 'test.STL'
resolution = 150

# ---------- Setup ----------
current_dir = Path(__file__).resolve().parent
input_path = current_dir / FileToRun
temp_dir = current_dir / "temp"
nukeTemp = True  # Set to False to keep temp files

FeatureList = []

# ---------- Safety Check ----------
if not input_path.exists():
    print(f"❌ Error: '{FileToRun}' does not exist.")
    exit()

# ---------- Create temp dir ----------
temp_dir.mkdir(exist_ok=True)

# ---------- Copy STL to temp ----------
temp_stl_path = temp_dir / FileToRun
shutil.copy(str(input_path), str(temp_stl_path))

# ---------- Convert STL to PNG ----------
conv.PreprocessSingleFile(temp_stl_path, resolution)

# ---------- Locate converted PNG ----------
png_filename = FileToRun.replace('.STL', '.png').replace('.stl', '.png')
temp_png_path = temp_dir / png_filename

# ---------- Run Preflight Check ----------
def RunPreflightCheck(png_path):
    if not png_path.exists():
        print(f"❌ Error: '{png_path}' does not exist.")
        return None
    output = PreflightCheck.predict_image(png_path)
    return output

# ---------- Run Reverse Engineering ----------
def RunReverseEngineeringModel(inputfile):
    return REM.ReverseEngineer(inputfile)

# ---------- Main Logic ----------
out = RunPreflightCheck(temp_png_path)

if out == 'CADmodel':
    print("The output is a CAD model.")
    FeatureList = RunReverseEngineeringModel(temp_png_path)
    print("Predicted Features:", FeatureList)

elif out == 'MESHmodel':
    print("This file cannot be reverse engineered into a CAD model.")

# ---------- Cleanup ----------
if temp_dir.exists() and nukeTemp:
    shutil.rmtree(temp_dir)
    print("🧹 Temp directory cleaned up.")
