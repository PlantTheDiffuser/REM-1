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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No file selected')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = UPLOAD_FOLDER / filename
            file.save(file_path)

            try:
                # Step 1: Copy STL to temp folder and preprocess
                TEMP_FOLDER.mkdir(exist_ok=True)
                temp_stl_path = TEMP_FOLDER / filename
                shutil.copy(str(file_path), str(temp_stl_path))
                conv.PreprocessSingleFile(temp_stl_path, resolution=150)

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
                # Save STL for front-end display
                static_stl_path = Path("static") / "uploaded_model.stl"
                shutil.copy(file_path, static_stl_path)


                # Clean up uploaded and temp files
                file_path.unlink(missing_ok=True)
                temp_stl_path.unlink(missing_ok=True)
                png_path.unlink(missing_ok=True)
                return render_template('index.html', features=features, result='✅ CAD Model detected.')

            except Exception as e:
                return render_template('index.html', error=f'Processing error: {e}')

        return render_template('index.html', error='Invalid file format')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
