from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
'''from skill_extraction import extract_skills_from_text
from utils import preprocess_resume
'''
app = Flask(__name__)
CORS(app)

# Serve the homepage (index.html)
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint to upload and process the resume
@app.route('/upload', methods=['POST'])
def upload_resume():
    try:
        file = request.files['pdf']
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        # Preprocess the resume
        resume_text = preprocess_resume(file)
        
        # Extract skills using the defined skill extraction method
        skills = extract_skills_from_text(resume_text)
        return jsonify({"skills": skills}), 200
    except Exception as e:
        return render_template('error.html', error_message=str(e))

# Error page route
@app.route('/error')
def error():
    return render_template('error.html', error_message="An error occurred. Please try again.")

if __name__ == "__main__":
    app.run(debug=True)
