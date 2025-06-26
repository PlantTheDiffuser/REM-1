import os
from pathlib import Path
import ConvertSTLtoVoxel as conv
import PreflightCheckTrainer

inputfile = 'model.STL'
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
    output = PreflightCheckTrainer.predict_image(input)
    #print("Output:", output)

    # Clean up temporary files
    input = Path(input)
    if input.exists():
        input.unlink()
        #print(f"🧹 Cleaned up temporary file: {input}")
    return output

out = RunPreflightCheck(inputfile)

