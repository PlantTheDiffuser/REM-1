# app.py
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from pathlib import Path
import shutil
import ConvertSTLtoVoxel as conv
import PreflightCheckTrainer as PreflightCheck
import ReverseEngineeringModel as REM

UPLOAD_FOLDER = Path("uploads")
TEMP_FOLDER = Path("temp")
ALLOWED_EXTENSIONS = {"stl"}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

UPLOAD_FOLDER.mkdir(exist_ok=True)
TEMP_FOLDER.mkdir(exist_ok=True)

STATIC_FOLDER = Path("static")
STATIC_FOLDER.mkdir(exist_ok=True)


def clear_old_uploads():
    """Remove previous uploads/temp files and app-generated static STLs.

    This removes files inside the `uploads` and `temp` folders and only
    removes the app-created STL files in `static` (uploaded_model.stl and
    working-*.stl). It avoids deleting other static assets.
    """
    # Remove files and directories from uploads and temp
    for folder in (UPLOAD_FOLDER, TEMP_FOLDER):
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        for p in folder.iterdir():
            try:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
            except Exception as e:
                # Log and continue; don't raise to avoid breaking request handling
                print(f"Warning: failed to remove {p}: {e}")

    # Remove only the STL files that the app creates in static
    try:
        static = STATIC_FOLDER
        for pattern in ("uploaded_model.stl", "working-*.stl"):
            for p in static.glob(pattern):
                try:
                    p.unlink()
                except Exception as e:
                    print(f"Warning: failed to remove static file {p}: {e}")
    except Exception as e:
        print(f"Warning while cleaning static files: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.form.get('reset'):
            # Clean up old files and return to the index
            clear_old_uploads()
            return render_template('index.html')

        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No file selected')
        if file and allowed_file(file.filename):
            # Remove previous uploads/temp/static STL files so each run is clean
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

                # Step 4: Run Reverse Engineering Model
                features = REM.ReverseEngineer(png_path)
                # Ensure static folder exists and save STL copies for front-end viewers
                static_dir = Path("static")
                static_dir.mkdir(exist_ok=True)

                # Save a master copy (for possible future use)
                static_stl_path = static_dir / "uploaded_model.stl"
                shutil.copy(file_path, static_stl_path)

                # The template expects 'working-1.stl', 'working-2.stl', ... in the static folder
                # Create one copy per predicted feature (skip the first entry which is the image path)
                for idx, _f in enumerate(features[1:], start=1):
                    target = static_dir / f"working-{idx}.stl"
                    # Copy the original STL so Three.js can load it for each viewer
                    shutil.copy(file_path, target)


                # Clean up uploaded and temp files
                file_path.unlink(missing_ok=True)
                temp_stl_path.unlink(missing_ok=True)
                png_path.unlink(missing_ok=True)
                return render_template('index.html', features=features, result='✅ CAD Model detected.')

            except Exception as e:
                return render_template('index.html', error=f'Processing error: {e}')

        return render_template('index.html', error='Invalid file format')

    return render_template('index.html')

@app.template_filter('basename_noext')
def basename_noext(path):
    import os
    return os.path.splitext(os.path.basename(path))[0]

if __name__ == '__main__':
    app.run(debug=True)
