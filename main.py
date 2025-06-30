from pathlib import Path
import ConvertSTLtoVoxel as conv
import PreflightCheckTrainer as PreflightCheck
import ReverseEngineeringModel as REM

FileToRun = 'test.STL'
current_dir = Path(__file__).resolve().parent

def RunPreflightCheck(inputfile):
    input_path = current_dir / inputfile
    if not input_path.exists():
        print(f"❌ Error: '{inputfile}' does not exist.")
        return None
    inputfilename = inputfile.split('.')[0]
    input = str(current_dir / inputfile)
    conv.PreprocessSingleFile(input, resolution=150)
    input = inputfilename + '.png'
    input = str(current_dir / input)
    output = PreflightCheck.predict_image(input)
    #print("Output:", output)

    # Clean up temporary files
    input = Path(input)
    if input.exists():
        input.unlink()
        #print(f"🧹 Cleaned up temporary file: {input}")
    return output

def RunReverseEngineeringModel(inputfile):
    REM.ReverseEngineer()

out = RunPreflightCheck(FileToRun)

if out == 'CADmodel':
    print("The output is a CAD model.")
    RunReverseEngineeringModel(FileToRun)

elif out == 'MESHmodel':
    print("This file cannot be reverse engineered into a CAD model.")