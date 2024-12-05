from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from skill_extraction import extract_skills

app = Flask(__name__)

# Upload folder and allowed extensions
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'backend', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pdf' not in request.files:
        return render_template('error.html', error_message='No file part provided.')

    file = request.files['pdf']
    if file.filename == '':
        return render_template('error.html', error_message='No file selected.')

    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Extract skills from the resume
        try:
            skills, experience, projects = extract_skills(filename)
            return render_template('skills.html', skills=skills, experience=experience, projects=projects)
        except Exception as e:
            return render_template('error.html', error_message=f"Error extracting skills: {str(e)}")
    else:
        return render_template('error.html', error_message='Invalid file type.')


@app.route('/mcq')
def mcq():
    return render_template('mcq.html')


if __name__ == '__main__':
    app.run(debug=True)
