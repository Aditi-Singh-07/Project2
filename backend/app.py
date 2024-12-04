from flask import Flask, request, jsonify
from flask_cors import CORS
from skill_extraction import extract_skills_from_text
from utils import preprocess_resume

app = Flask(__name__)
CORS(app)

@app.route('/upload', methods=['POST'])
def upload_resume():
    try:
        file = request.files['pdf']
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        # Preprocess the resume
        resume_text = preprocess_resume(file)
        skills = extract_skills_from_text(resume_text)
        return jsonify({"skills": skills}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
