# app.py
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from pathlib import Path
import shutil
from PIL import Image

import ConvertSTLtoVoxel as conv
import PreflightCheckTrainer as PreflightCheck
import ReverseEngineeringModel as REM  # <-- new model lives here

UPLOAD_FOLDER = Path("uploads")
TEMP_FOLDER = Path("temp")
STATIC_FOLDER = Path("static")
ALLOWED_EXTENSIONS = {"stl"}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

for folder in (UPLOAD_FOLDER, TEMP_FOLDER, STATIC_FOLDER):
    folder.mkdir(exist_ok=True)

# ---------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------

def clear_old_uploads():
    """Remove previous uploads/temp files and app-generated static STLs."""
    for folder in (UPLOAD_FOLDER, TEMP_FOLDER):
        folder.mkdir(parents=True, exist_ok=True)
        for p in folder.iterdir():
            try:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
            except Exception as e:
                print(f"Warning: failed to remove {p}: {e}")

    try:
        for pattern in ("uploaded_model.stl", "working-*.stl"):
            for p in STATIC_FOLDER.glob(pattern):
                try:
                    p.unlink()
                except Exception as e:
                    print(f"Warning: failed to remove static file {p}: {e}")
    except Exception as e:
        print(f"Warning while cleaning static files: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------------------------------------------------------
# Main route
# ---------------------------------------------------------------

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.form.get('reset'):
            clear_old_uploads()
            return render_template('index.html')

        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No file selected')

        if file and allowed_file(file.filename):
            clear_old_uploads()
            filename = secure_filename(file.filename)
            file_path = UPLOAD_FOLDER / filename
            file.save(file_path)

            try:
                # Step 1: Copy STL to temp folder and preprocess
                TEMP_FOLDER.mkdir(exist_ok=True)
                temp_stl_path = TEMP_FOLDER / filename
                shutil.copy(str(file_path), str(temp_stl_path))
                conv.PreprocessSingleFile(temp_stl_path, resolution=150, cleanup=True)

                # Step 2: Locate PNG output
                png_filename = filename.replace('.STL', '.png').replace('.stl', '.png')
                png_path = TEMP_FOLDER / png_filename
                if not png_path.exists():
                    return render_template('index.html', error='PNG conversion failed.')

                # Step 3: Run Preflight check
                result = PreflightCheck.predict_image(png_path)
                if result == 'MESHmodel':
                    return render_template('index.html', result='❌ File is a mesh model and cannot be reverse engineered.')

                # ensure working image exists
                working_path = png_path.parent / "working.png"
                if not working_path.exists():
                    Image.new("L", (150, 150**2), color=0).save(working_path)
                
                # Step 4: Run Reverse Engineering Model (classification)
                features = [str(png_path)]

                try:
                    # explicitly pass path to FeatureClassifier.pth
                    model_path = Path(__file__).resolve().parent / "FeatureClassifier.pth"
                    prediction = REM.classify_feature(working_path, png_path, model_path=model_path)
                    features.append(prediction["top1"])
                except Exception as e:
                    return render_template('index.html', error=f'Classification error: {e}')

                # Step 5: Prepare static STL files for viewer
                static_stl_path = STATIC_FOLDER / "uploaded_model.stl"
                shutil.copy(file_path, static_stl_path)

                for idx, _f in enumerate(features[1:], start=1):
                    target = STATIC_FOLDER / f"working-{idx}.stl"
                    shutil.copy(file_path, target)

                # Clean up
                file_path.unlink(missing_ok=True)
                temp_stl_path.unlink(missing_ok=True)
                png_path.unlink(missing_ok=True)

                return render_template('index.html', features=features, result='✅ CAD Model detected.')

            except Exception as e:
                return render_template('index.html', error=f'Processing error: {e}')

        return render_template('index.html', error='Invalid file format')

    return render_template('index.html')

# ---------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------

@app.template_filter('basename_noext')
def basename_noext(path):
    import os
    return os.path.splitext(os.path.basename(path))[0]

# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True)
