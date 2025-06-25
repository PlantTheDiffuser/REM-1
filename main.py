import os
from pathlib import Path
import PreflightCheckTrainer

current_dir = Path(__file__).resolve().parent
inputImage = str(current_dir / 'Umbral_Revenant_Miniature.png')

output = PreflightCheckTrainer.predict_image(inputImage)
print("Output:", output)
