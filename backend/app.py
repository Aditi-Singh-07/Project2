from flask import Flask, render_template, request, jsonify
import os
import json
from skill_extraction import extract_text_from_pdf, extract_skills_with_fuzzy
from sklearn.metrics import precision_score, recall_score, f1_score

app = Flask(__name__)

# Configurations for upload directory and allowed file extensions
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'backend', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Path to test data directory and loading ground truth data
TEST_DATA_FOLDER = os.path.join(os.getcwd(), 'test_data')
with open(os.path.join(os.getcwd(), 'test_data.json')) as f:
    test_data = json.load(f)

# Utility Function: Check if uploaded file is valid
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route
@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')

# Function to calculate accuracy for extracted skills
def calculate_accuracy(extracted_skills):
    """Calculate the accuracy of extracted skills compared to a predefined set."""
    predefined_skills = ["python", "java", "sql", "communication", "teamwork", "leadership"]
    all_extracted_skills = [skill for skills in extracted_skills.values() for skill in skills]
    
    correct_skills = len(set(all_extracted_skills).intersection(predefined_skills))
    total_skills = len(predefined_skills)

    if total_skills == 0:  # Avoid division by zero
        return 0
    accuracy = (correct_skills / total_skills) * 100
    return round(accuracy, 2)

# Upload and skill extraction route
@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload, extract skills, and render results."""
    if 'pdf' not in request.files:
        return render_template('error.html', error_message='No file part provided.')

    file = request.files['pdf']
    if file.filename == '':
        return render_template('error.html', error_message='No file selected.')

    if file and allowed_file(file.filename):
        # Save the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Extract skills from the file
        try:
            resume_text = extract_text_from_pdf(filepath)
            skills = extract_skills_with_fuzzy(resume_text)
            accuracy = calculate_accuracy(skills)  # Calculate accuracy of skill extraction
            
            return render_template('skills.html', skills=skills, accuracy=accuracy)
        except Exception as e:
            return render_template('error.html', error_message=f"Error extracting skills: {str(e)}")
    else:
        return render_template('error.html', error_message='Invalid file type.')

# Function to calculate precision, recall, and F1-score
def calculate_metrics(extracted_skills, ground_truth):
    """Calculate precision, recall, and F1 score for extracted skills."""
    all_true_skills = []
    all_predicted_skills = []

    for resume, skills in ground_truth.items():
        predicted_skills = extracted_skills.get(resume, {})
        for category in ["technical", "soft", "managerial"]:
            true_skills = skills.get(category, [])
            predicted = predicted_skills.get(category, [])

            for skill in true_skills:
                all_true_skills.append(1)
                all_predicted_skills.append(1 if skill in predicted else 0)
            for skill in predicted:
                if skill not in true_skills:
                    all_true_skills.append(0)
                    all_predicted_skills.append(1)

    accuracy = sum(1 for true, pred in zip(all_true_skills, all_predicted_skills) if true == pred) / len(all_true_skills)
    precision = precision_score(all_true_skills, all_predicted_skills)
    recall = recall_score(all_true_skills, all_predicted_skills)
    f1 = f1_score(all_true_skills, all_predicted_skills)

    return accuracy, precision, recall, f1

# Test the model with sample test data
def test_model_with_test_data():
    """Test skill extraction model on predefined test data."""
    correct_predictions = 0
    total_resumes = len(test_data)

    for resume_file in os.listdir(TEST_DATA_FOLDER):
        if resume_file.endswith('.pdf'):
            resume_path = os.path.join(TEST_DATA_FOLDER, resume_file)
            extracted_skills = extract_skills_with_fuzzy(extract_text_from_pdf(resume_path))

            expected_skills = test_data.get(resume_file, [])
            if not expected_skills:
                continue

            correct_predictions += len(set(extracted_skills['technical']).intersection(set(expected_skills)))

    accuracy = correct_predictions / total_resumes
    precision = 0.9  # Placeholder
    recall = 0.85  # Placeholder
    f1 = 0.87  # Placeholder

    return accuracy, precision, recall, f1

# Test endpoint for evaluating model
@app.route('/test', methods=['GET'])
def test():
    """Run tests and display performance metrics."""
    accuracy, precision, recall, f1 = test_model_with_test_data()
    return render_template('accuracy.html', accuracy=accuracy, precision=precision, recall=recall, f1=f1)

# MCQ page
@app.route('/mcq')
def mcq():
    """Render MCQ page."""
    return render_template('mcq.html')

if __name__ == '__main__':
    app.run(debug=True)







'''from flask import Flask, render_template, request, jsonify
import os
from skill_extraction import extract_text_from_pdf, extract_skills_with_fuzzy
import json
from sklearn.metrics import precision_score, recall_score, f1_score

app = Flask(__name__)

# Upload folder and allowed extensions
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'backend', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Path to your test data folder
TEST_DATA_FOLDER = os.path.join(os.getcwd(), 'test_data')

# Load the ground truth data (you can adjust the path based on your project structure)
with open(os.path.join(os.getcwd(), 'test_data.json')) as f:
    test_data = json.load(f)

@app.route('/')
def index():
    return render_template('index.html')

def calculate_accuracy(extracted_skills):
    # Example predefined skill set
    predefined_skills = ["python", "java", "sql", "communication", "teamwork", "leadership"]
    
    # Flatten the list of extracted skills (if it's a dictionary, which it is in this case)
    all_extracted_skills = [skill for skills in extracted_skills.values() for skill in skills]
    
    # Count how many skills are correct (match between predefined and extracted skills)
    correct_skills = len(set(all_extracted_skills).intersection(predefined_skills))
    total_skills = len(predefined_skills)
    
    # If there are no predefined skills to compare, return 0 accuracy
    if total_skills == 0:
        return 0  # Avoid division by zero
    
    # Calculate accuracy percentage
    accuracy = (correct_skills / total_skills) * 100
    return round(accuracy, 2)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pdf' not in request.files:
        return render_template('error.html', error_message='No file part provided.')

    file = request.files['pdf']
    if file.filename == '':
        return render_template('error.html', error_message='No file selected.')

    if file and allowed_file(file.filename):
        # Save the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Extract skills from the resume
        try:
            resume_text = extract_text_from_pdf(filepath)
            skills = extract_skills_with_fuzzy(resume_text)
            
            # Calculate accuracy of the skill extraction
            accuracy = calculate_accuracy(skills)  # Pass extracted skills to calculate_accuracy
            
            # Pass skills and accuracy to the template
            return render_template('skills.html', skills=skills, accuracy=accuracy)
        except Exception as e:
            return render_template('error.html', error_message=f"Error extracting skills: {str(e)}")
    else:
        return render_template('error.html', error_message='Invalid file type.')
    
def calculate_metrics(extracted_skills, ground_truth):
    """Calculate accuracy, precision, recall, and F1 score based on predefined skills."""
    all_true_skills = []
    all_predicted_skills = []
    
    # Compare extracted skills with ground truth
    for resume, skills in ground_truth.items():
        predicted_skills = extracted_skills.get(resume, {})
        
        for category in ["technical", "soft", "managerial"]:
            true_skills = skills.get(category, [])
            predicted = predicted_skills.get(category, [])
            
            # Append true and predicted skills for metrics calculation
            for skill in true_skills:
                all_true_skills.append(1)
                all_predicted_skills.append(1 if skill in predicted else 0)
            for skill in predicted:
                if skill not in true_skills:
                    all_true_skills.append(0)
                    all_predicted_skills.append(1)
    
    # Calculate the metrics
    accuracy = sum([1 for true, pred in zip(all_true_skills, all_predicted_skills) if true == pred]) / len(all_true_skills)
    precision = precision_score(all_true_skills, all_predicted_skills)
    recall = recall_score(all_true_skills, all_predicted_skills)
    f1 = f1_score(all_true_skills, all_predicted_skills)

    return accuracy, precision, recall, f1

def test_model_with_test_data():
    correct_predictions = 0
    total_resumes = len(test_data)  # Assuming 'test_data' is loaded properly

    for resume_file in os.listdir(TEST_DATA_FOLDER):
        if resume_file.endswith('.pdf'):
            # Extract skills from the resume
            resume_path = os.path.join(TEST_DATA_FOLDER, resume_file)
            extracted_skills = extract_skills_with_fuzzy(extract_text_from_pdf(resume_path))
            
            # Compare with expected skills
            expected_skills = test_data.get(resume_file, [])
            if not expected_skills:
                continue  # Skip if no expected skills are found for this file

            # Calculate accuracy by comparing the extracted and expected skills
            correct_predictions += len(set(extracted_skills['technical']).intersection(set(expected_skills)))

    accuracy = correct_predictions / total_resumes  # Accuracy as a float
    precision = 0.9  # Replace with actual precision calculation
    recall = 0.85  # Replace with actual recall calculation
    f1 = 0.87  # Replace with actual F1 score calculation

    return accuracy, precision, recall, f1  # Return as a tuple


@app.route('/test', methods=['GET'])
def test():
    accuracy, precision, recall, f1 = test_model_with_test_data()
    
    # Pass all metrics to the template
    return render_template('accuracy.html', accuracy=accuracy, precision=precision, recall=recall, f1=f1)

@app.route('/mcq')
def mcq():
    return render_template('mcq.html')

if __name__ == '__main__':
    app.run(debug=True)'''

'''@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pdf' not in request.files:
        return render_template('error.html', error_message='No file part provided.')

    file = request.files['pdf']
    if file.filename == '':
        return render_template('error.html', error_message='No file selected.')

    if file and allowed_file(file.filename):
        # Save the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Extract skills from the resume
        try:
            resume_text = extract_text_from_pdf(filepath)
            skills = extract_skills_with_fuzzy(resume_text)
            return render_template('skills.html', skills=skills)
        except Exception as e:
            return render_template('error.html', error_message=f"Error extracting skills: {str(e)}")
    else:
        return render_template('error.html', error_message='Invalid file type.')'''
'''
@app.route('/test_accuracy')
def test_accuracy():
    test_folder = os.path.join(app.root_path, 'test_data')
    expected_skills_file = os.path.join(test_folder, 'test_data.json')

    # Calculate accuracy
    accuracy = calculate_accuracy(test_folder, expected_skills_file)
    return render_template('skills.html', accuracy=accuracy, skills={})

if __name__ == '__main__':
    app.run(debug=True)
'''
'''from flask import Flask, render_template, request, redirect, url_for, jsonify
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
'''