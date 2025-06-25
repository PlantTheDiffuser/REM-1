import os
from pathlib import Path
import ConvertSTLtoVoxel as conv
import PreflightCheckTrainer

inputfile = 'test.STL'
inputfilename = inputfile.split('.')[0]

current_dir = Path(__file__).resolve().parent
input = str(current_dir / inputfile)
conv.PreprocessSingleFile(input, resolution=150)


input = inputfilename + '.png'
input = str(current_dir / input)
output = PreflightCheckTrainer.predict_image(input)
print("Output:", output)

# Clean up temporary files
input = Path(input)
if input.exists():
    input.unlink()
    #print(f"🧹 Cleaned up temporary file: {input}")