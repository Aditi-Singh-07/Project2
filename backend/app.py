from flask import Flask, render_template, request, jsonify
import os
import json
from skill_extraction import extract_text_from_pdf, extract_skills_with_fuzzy
from sklearn.metrics import precision_score, recall_score, f1_score
import requests

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

# Predefined skill ontology (for eligibility check)
SKILL_ONTOLOGY = {
    "technical": ["django", "python", "sql", "java", "tensorflow", "html", "mysql", "javascript", "git", "css"],
    "soft": ["communication", "teamwork", "problem-solving"],
    "managerial": ["leadership", "strategic planning", "management"]
}

# Trivia API endpoint
TRIVIA_API_URL = "https://opentdb.com/api.php"

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
    predefined_skills = [skill.lower() for skill in SKILL_ONTOLOGY["technical"] + SKILL_ONTOLOGY["soft"] + SKILL_ONTOLOGY["managerial"]]
    all_extracted_skills = [skill.lower() for skills in extracted_skills.values() for skill in skills]

    correct_skills = len(set(all_extracted_skills).intersection(predefined_skills))
    total_skills = len(predefined_skills)

    return round((correct_skills / total_skills) * 100, 2) if total_skills > 0 else 0

# Updated eligibility logic to allow at least 2 skills to match
def check_eligibility(extracted_skills):
    total_matching_skills = 0

    # Check technical skills category for at least 2 matching skills
    technical_skills = SKILL_ONTOLOGY.get('technical', [])
    matching_technical_skills = set(extracted_skills.get('technical', []))
    total_matching_skills += len(matching_technical_skills)

    # If at least 2 skills match in the technical category, they are eligible
    if total_matching_skills >= 2:
        return True
    else:
        return False

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
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            resume_text = extract_text_from_pdf(filepath)
            skills = extract_skills_with_fuzzy(resume_text)
            accuracy = calculate_accuracy(skills)
            eligible = check_eligibility(skills)

            return render_template('skills.html', skills=skills, accuracy=accuracy, eligible=eligible)
        except Exception as e:
            return render_template('error.html', error_message=f"Error extracting skills: {str(e)}")
    else:
        return render_template('error.html', error_message='Invalid file type.')

# Generate MCQs using Open Trivia Database API
def generate_mcqs(extracted_skills):
    """Generate MCQs based on extracted skills using an external API."""
    mcqs = {"technical": [], "soft": [], "managerial": []}
    api_url = "https://opentdb.com/api.php"

    total_questions = 20
    question_distribution = {
        "technical": int(total_questions * 0.6),
        "soft": int(total_questions * 0.2),
        "managerial": int(total_questions * 0.2)
    }

    for category, count in question_distribution.items():
        response = requests.get(api_url, params={
            "amount": count,
            "type": "multiple",
            "difficulty": "medium"
        })
        if response.status_code == 200:
            questions = response.json().get('results', [])
            for question in questions:
                mcqs[category].append({
                    "id": f"{category}_{len(mcqs[category])}",
                    "question": question['question'],
                    "options": question['incorrect_answers'] + [question['correct_answer']],
                    "answer": question['correct_answer']
                })

    # Shuffle the options for each question
    for category_questions in mcqs.values():
        for question in category_questions:
            question['options'] = sorted(question['options'])

    return mcqs

# Route to start the test and display MCQs
@app.route('/start_test', methods=['GET'])
def start_test():
    """Generate MCQs and display the test page."""
    extracted_skills = {
        'technical': ['python', 'java'],
        'soft': ['communication'],
        'managerial': ['leadership']
    }
    mcqs = generate_mcqs(extracted_skills)
    return render_template('test.html', mcqs=mcqs, duration=30)

# Submit test and calculate score
@app.route('/submit_test', methods=['POST'])
def submit_test():
    """Calculate and display the score."""
    submitted_answers = request.json.get('answers', {})
    correct_answers = request.json.get('correct_answers', {})

    score = 0
    for category, questions in correct_answers.items():
        for qid, correct_answer in questions.items():
            if submitted_answers.get(qid) == correct_answer:
                score += 1

    return jsonify({"score": score, "total": len(correct_answers)})

# Test the model with sample test data
def test_model_with_test_data():
    """Test skill extraction model on predefined test data."""
    correct_predictions = 0
    total_skills = 0

    for resume_file, ground_truth_skills in test_data.items():
        resume_path = os.path.join(TEST_DATA_FOLDER, resume_file)
        extracted_skills = extract_skills_with_fuzzy(extract_text_from_pdf(resume_path))

        for category in ['technical', 'soft', 'managerial']:
            true_skills = set(ground_truth_skills.get(category, []))
            predicted_skills = set(extracted_skills.get(category, []))

            correct_predictions += len(true_skills.intersection(predicted_skills))
            total_skills += len(true_skills)

    accuracy = round((correct_predictions / total_skills) * 100, 2) if total_skills > 0 else 0
    precision = precision_score([1] * correct_predictions + [0] * (total_skills - correct_predictions),
                                 [1] * correct_predictions + [0] * (total_skills - correct_predictions))
    recall = recall_score([1] * correct_predictions + [0] * (total_skills - correct_predictions),
                           [1] * correct_predictions + [0] * (total_skills - correct_predictions))
    f1 = f1_score([1] * correct_predictions + [0] * (total_skills - correct_predictions),
                  [1] * correct_predictions + [0] * (total_skills - correct_predictions))

    return accuracy, precision, recall, f1

# Test endpoint for evaluating model
@app.route('/test', methods=['GET'])
def test():
    """Run tests and display performance metrics."""
    accuracy, precision, recall, f1 = test_model_with_test_data()
    return render_template('accuracy.html', accuracy=accuracy, precision=precision, recall=recall, f1=f1)

# Eligibility route to show eligibility status
@app.route('/eligibility')
def eligibility_check():
    """Render the eligibility check page."""
    eligible = False
    # Example: Assume extracted_skills are passed through context or session
    extracted_skills = {
        'technical': ['python', 'java', 'html'],
        'soft': ['communication', 'teamwork'],
        'managerial': ['leadership', 'management']
    }
    eligible = check_eligibility(extracted_skills)
    return render_template('eligibility.html', eligible=eligible)

@app.route('/mcq')
def mcq():
    """Render MCQ page."""
    return render_template('mcq.html')

@app.route('/instructions')
def instructions():
    """Display instructions for the test."""
    return render_template('instructions.html')

if __name__ == '__main__':
    app.run(debug=True)










'''from flask import Flask, render_template, request, jsonify
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

# Predefined skill ontology (for eligibility check)
SKILL_ONTOLOGY = {
    "technical": ["django", "python", "sql", "java", "tensorflow", "html", "mysql", "javascript", "git", "css"],
    "soft": ["communication", "teamwork", "problem-solving"],
    "managerial": ["leadership", "strategic planning", "management"]
}

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
    predefined_skills = [skill.lower() for skill in SKILL_ONTOLOGY["technical"] + SKILL_ONTOLOGY["soft"] + SKILL_ONTOLOGY["managerial"]]
    all_extracted_skills = [skill.lower() for skills in extracted_skills.values() for skill in skills]

    correct_skills = len(set(all_extracted_skills).intersection(predefined_skills))
    total_skills = len(predefined_skills)

    return round((correct_skills / total_skills) * 100, 2) if total_skills > 0 else 0

# Updated eligibility logic to allow at least 2 skills to match
def check_eligibility(extracted_skills):
    total_matching_skills = 0

    # Check technical skills category for at least 2 matching skills
    technical_skills = SKILL_ONTOLOGY.get('technical', [])
    matching_technical_skills = set(extracted_skills.get('technical', []))
    total_matching_skills += len(matching_technical_skills)

    # If at least 2 skills match in the technical category, they are eligible
    if total_matching_skills >= 2:
        return True
    else:
        return False

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
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            resume_text = extract_text_from_pdf(filepath)
            skills = extract_skills_with_fuzzy(resume_text)
            accuracy = calculate_accuracy(skills)
            eligible = check_eligibility(skills)

            return render_template('skills.html', skills=skills, accuracy=accuracy, eligible=eligible)
        except Exception as e:
            return render_template('error.html', error_message=f"Error extracting skills: {str(e)}")
    else:
        return render_template('error.html', error_message='Invalid file type.')

# Test the model with sample test data
def test_model_with_test_data():
    """Test skill extraction model on predefined test data."""
    correct_predictions = 0
    total_skills = 0

    for resume_file, ground_truth_skills in test_data.items():
        resume_path = os.path.join(TEST_DATA_FOLDER, resume_file)
        extracted_skills = extract_skills_with_fuzzy(extract_text_from_pdf(resume_path))

        for category in ['technical', 'soft', 'managerial']:
            true_skills = set(ground_truth_skills.get(category, []))
            predicted_skills = set(extracted_skills.get(category, []))

            correct_predictions += len(true_skills.intersection(predicted_skills))
            total_skills += len(true_skills)

    accuracy = round((correct_predictions / total_skills) * 100, 2) if total_skills > 0 else 0
    precision = precision_score([1] * correct_predictions + [0] * (total_skills - correct_predictions),
                                 [1] * correct_predictions + [0] * (total_skills - correct_predictions))
    recall = recall_score([1] * correct_predictions + [0] * (total_skills - correct_predictions),
                           [1] * correct_predictions + [0] * (total_skills - correct_predictions))
    f1 = f1_score([1] * correct_predictions + [0] * (total_skills - correct_predictions),
                  [1] * correct_predictions + [0] * (total_skills - correct_predictions))

    return accuracy, precision, recall, f1

# Test endpoint for evaluating model
@app.route('/test', methods=['GET'])
def test():
    """Run tests and display performance metrics."""
    accuracy, precision, recall, f1 = test_model_with_test_data()
    return render_template('accuracy.html', accuracy=accuracy, precision=precision, recall=recall, f1=f1)

# Eligibility route to show eligibility status
@app.route('/eligibility')
def eligibility_check():
    """Render the eligibility check page."""
    eligible = False
    # Example: Assume extracted_skills are passed through context or session
    extracted_skills = {
        'technical': ['python', 'java', 'html'],
        'soft': ['communication', 'teamwork'],
        'managerial': ['leadership', 'management']
    }
    eligible = check_eligibility(extracted_skills)
    return render_template('eligibility.html', eligible=eligible)

@app.route('/mcq')
def mcq():
    """Render MCQ page."""
    return render_template('mcq.html')

@app.route('/instructions')
def instructions():
    """Display instructions for the test."""
    return render_template('instructions.html')

if __name__ == '__main__':
    app.run(debug=True)'''








'''from flask import Flask, render_template, request, jsonify
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

# Predefined skill ontology (for eligibility check)
SKILL_ONTOLOGY = {
    "technical": ["django", "python", "sql", "java", "tensorflow", "html", "mysql", "javascript", "git", "css"],
    "soft": ["communication", "teamwork", "problem-solving"],
    "managerial": ["leadership", "strategic planning", "management"]
}

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
    predefined_skills = ["java", "sql", "django", "python", "mysql", "javascript", "git", "html", "css", "communication", "problem-solving", "teamwork", "management"]
    all_extracted_skills = [skill for skills in extracted_skills.values() for skill in skills]
    
    correct_skills = len(set(all_extracted_skills).intersection(predefined_skills))
    total_skills = len(predefined_skills)

    if total_skills == 0:  # Avoid division by zero
        return 0
    accuracy = (correct_skills / total_skills) * 100
    return round(accuracy, 2)

# Function to check eligibility based on 30% match in each skill category
# Function to check eligibility based on 30% match in technical skills
def check_eligibility(extracted_skills):
    eligible = True

    # Check technical skills category for 30% match
    technical_skills = SKILL_ONTOLOGY.get('technical', [])
    matching_technical_skills = set(extracted_skills.get('technical', []))
    technical_match_percentage = (len(matching_technical_skills) / len(technical_skills)) * 100

    # If technical skills match is less than 30%, set eligible to False
    if technical_match_percentage < 30:
        eligible = False
    
    # Check other categories (soft and managerial) without the 30% match restriction
    if eligible:  # If still eligible, check soft and managerial skills
        for category in ['soft', 'managerial']:
            skills = SKILL_ONTOLOGY.get(category, [])
            matching_skills = set(extracted_skills.get(category, []))
            match_percentage = (len(matching_skills) / len(skills)) * 100

            # If any category has a match below 30%, eligibility fails
            if match_percentage < 30:
                eligible = False
                break
    
    return eligible
'''


'''def check_eligibility(extracted_skills):
    eligible = True
    for category, skills in SKILL_ONTOLOGY.items():
        # Normalize skills to lowercase for consistent matching
        extracted_category_skills = {skill.lower() for skill in extracted_skills.get(category, [])}
        predefined_category_skills = {skill.lower() for skill in skills}

        # Calculate the number of matching skills
        matching_skills = extracted_category_skills.intersection(predefined_category_skills)

        # Calculate the match percentage
        match_percentage = (len(matching_skills) / len(predefined_category_skills)) * 100
        print(f"Category: {category}, Matching Skills: {matching_skills}, Match Percentage: {match_percentage}%")
        
        if match_percentage < 30:
            eligible = False
            break

    return eligible'''

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
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            resume_text = extract_text_from_pdf(filepath)
            skills = extract_skills_with_fuzzy(resume_text)
            accuracy = calculate_accuracy(skills)

            # Check eligibility
            eligible = check_eligibility(skills)

            return render_template('skills.html', skills=skills, accuracy=accuracy, eligible=eligible)
        except Exception as e:
            return render_template('error.html', error_message=f"Error extracting skills: {str(e)}")
    else:
        return render_template('error.html', error_message='Invalid file type.')

# Test the model with sample test data
def test_model_with_test_data():
    """Test skill extraction model on predefined test data."""
    correct_predictions = 0
    total_skills = 0

    for resume_file, ground_truth_skills in test_data.items():
        resume_path = os.path.join(TEST_DATA_FOLDER, resume_file)
        extracted_skills = extract_skills_with_fuzzy(extract_text_from_pdf(resume_path))

        for category in ['technical', 'soft', 'managerial']:
            true_skills = set(ground_truth_skills.get(category, []))
            predicted_skills = set(extracted_skills.get(category, []))

            correct_predictions += len(true_skills.intersection(predicted_skills))
            total_skills += len(true_skills)

    accuracy = round((correct_predictions / total_skills) * 100, 2) if total_skills > 0 else 0
    precision = precision_score([1] * correct_predictions + [0] * (total_skills - correct_predictions),
                                 [1] * correct_predictions + [0] * (total_skills - correct_predictions))
    recall = recall_score([1] * correct_predictions + [0] * (total_skills - correct_predictions),
                           [1] * correct_predictions + [0] * (total_skills - correct_predictions))
    f1 = f1_score([1] * correct_predictions + [0] * (total_skills - correct_predictions),
                  [1] * correct_predictions + [0] * (total_skills - correct_predictions))

    return accuracy, precision, recall, f1

# Test endpoint for evaluating model
@app.route('/test', methods=['GET'])
def test():
    """Run tests and display performance metrics."""
    accuracy, precision, recall, f1 = test_model_with_test_data()
    return render_template('accuracy.html', accuracy=accuracy, precision=precision, recall=recall, f1=f1)

@app.route('/mcq')
def mcq():
    """Render MCQ page."""
    return render_template('mcq.html')

@app.route('/eligibility')
def eligibility_check():
    """Render the eligibility check page."""
    # Just return a placeholder page or logic based on your requirements
    return render_template('eligibility.html')

if __name__ == '__main__':
    app.run(debug=True)






'''from flask import Flask, render_template, request, jsonify
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
    predefined_skills = ["java", "sql", "django", "python", "mysql", "javascript", "git", "html", "css", "communication", "problem-solving", "teamwork", "management"]
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
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            resume_text = extract_text_from_pdf(filepath)
            skills = extract_skills_with_fuzzy(resume_text)
            accuracy = calculate_accuracy(skills)
            return render_template('skills.html', skills=skills, accuracy=accuracy)
        except Exception as e:
            return render_template('error.html', error_message=f"Error extracting skills: {str(e)}")
    else:
        return render_template('error.html', error_message='Invalid file type.')



# Test the model with sample test data
def test_model_with_test_data():
    """Test skill extraction model on predefined test data."""
    correct_predictions = 0
    total_skills = 0

    for resume_file, ground_truth_skills in test_data.items():
        resume_path = os.path.join(TEST_DATA_FOLDER, resume_file)
        extracted_skills = extract_skills_with_fuzzy(extract_text_from_pdf(resume_path))

        for category in ['technical', 'soft', 'managerial']:
            true_skills = set(ground_truth_skills.get(category, []))
            predicted_skills = set(extracted_skills.get(category, []))

            correct_predictions += len(true_skills.intersection(predicted_skills))
            total_skills += len(true_skills)

    accuracy = round((correct_predictions / total_skills) * 100,2) if total_skills > 0 else 0
    precision = precision_score([1] * correct_predictions + [0] * (total_skills - correct_predictions),
                                 [1] * correct_predictions + [0] * (total_skills - correct_predictions))
    recall = recall_score([1] * correct_predictions + [0] * (total_skills - correct_predictions),
                           [1] * correct_predictions + [0] * (total_skills - correct_predictions))
    f1 = f1_score([1] * correct_predictions + [0] * (total_skills - correct_predictions),
                  [1] * correct_predictions + [0] * (total_skills - correct_predictions))

    return accuracy, precision, recall, f1

# Test endpoint for evaluating model
@app.route('/test', methods=['GET'])
def test():
    """Run tests and display performance metrics."""
    accuracy, precision, recall, f1 = test_model_with_test_data()
    return render_template('accuracy.html', accuracy=accuracy, precision=precision, recall=recall, f1=f1)

@app.route('/mcq')
def mcq():
    """Render MCQ page."""
    return render_template('mcq.html')

if __name__ == '__main__':
    app.run(debug=True)'''






'''from flask import Flask, render_template, request, jsonify
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
    predefined_skills = ["java", "sql", "django", "python","mysql","javascript","git","html","css", "communication", "problem-solving", "teamwork", "management"]
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
    total_skills = 0

    for resume_file, ground_truth_skills in test_data.items():
        resume_path = os.path.join(TEST_DATA_FOLDER, resume_file)
        extracted_skills = extract_skills_with_fuzzy(extract_text_from_pdf(resume_path))

        # If ground_truth_skills is a list, iterate over all categories
        for category in ['technical', 'soft', 'managerial']:
            true_skills = set(ground_truth_skills)  # Directly treat it as a list of skills
            predicted_skills = set(extracted_skills.get(category, []))  # Default to empty list if category not found

            # Count correct matches
            correct_predictions += len(true_skills.intersection(predicted_skills))
            total_skills += len(true_skills)

    # Ensure we do not divide by zero
    if total_skills == 0:
        accuracy = 0
    else:
        accuracy = (correct_predictions / total_skills) * 100

    precision = 0.9  # Placeholder for precision calculation
    recall = 0.85  # Placeholder for recall calculation
    f1 = 0.87  # Placeholder for f1-score calculation

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
    app.run(debug=True)'''

'''from flask import Flask, render_template, request, jsonify
import os
import json
from skill_extraction import extract_text_from_pdf, extract_skills_with_fuzzy, calculate_accuracy
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
            accuracy = calculate_accuracy(skills, test_data)  # Calculate accuracy of skill extraction
            
            return render_template('skills.html', skills=skills, accuracy=accuracy)
        except Exception as e:
            return render_template('error.html', error_message=f"Error extracting skills: {str(e)}")
    else:
        return render_template('error.html', error_message='Invalid file type.')

# Test the model with sample test data
def test_model_with_test_data():
    """Test skill extraction model on predefined test data."""
    correct_predictions = 0
    total_resumes = len(test_data)
    total_ground_truth_skills = 0

    for resume_file, ground_truth_skills in test_data.items():
        # Assuming the resume_file is a PDF file in TEST_DATA_FOLDER
        resume_path = os.path.join(TEST_DATA_FOLDER, resume_file)
        extracted_skills = extract_skills_with_fuzzy(extract_text_from_pdf(resume_path))

        # Compare extracted skills with ground truth for each category
        for category in ['technical', 'soft', 'managerial']:
            true_skills = ground_truth_skills.get(category, [])
            predicted_skills = extracted_skills.get(category, [])
            
            correct_predictions += len(set(true_skills).intersection(set(predicted_skills)))
            total_ground_truth_skills += len(true_skills)

    # Avoid division by zero
    if total_ground_truth_skills == 0:
        accuracy = 0
    else:
        accuracy = (correct_predictions / total_ground_truth_skills) * 100

    precision = 0.9  # Placeholder for actual precision calculation
    recall = 0.85  # Placeholder for actual recall calculation
    f1 = 0.87  # Placeholder for actual f1-score calculation

    return accuracy, precision, recall, f1'''

'''def test_model_with_test_data():
    """Test skill extraction model on predefined test data."""
    correct_predictions = 0
    total_resumes = len(test_data)

    for resume_file, ground_truth_skills in test_data.items():
        # Assuming the resume_file is a PDF file in TEST_DATA_FOLDER
        resume_path = os.path.join(TEST_DATA_FOLDER, resume_file)
        extracted_skills = extract_skills_with_fuzzy(extract_text_from_pdf(resume_path))

        # Comparing extracted skills with the ground truth
        correct_predictions += len(set(extracted_skills['technical']).intersection(set(ground_truth_skills)))

    accuracy = (correct_predictions / total_resumes) * 100
    precision = 0.9  # Placeholder for actual precision calculation
    recall = 0.85  # Placeholder for actual recall calculation
    f1 = 0.87  # Placeholder for actual f1-score calculation

    return accuracy, precision, recall, f1'''

# Test endpoint for evaluating model
@app.route('/test', methods=['GET'])
def test():
    """Run tests and display performance metrics."""
    accuracy, precision, recall, f1 = test_model_with_test_data()
    return render_template('accuracy.html', accuracy=accuracy, precision=precision, recall=recall, f1=f1)

if __name__ == '__main__':
    app.run(debug=True)



'''from flask import Flask, render_template, request, jsonify
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

    # Calculate metrics using sklearn
    precision = precision_score(all_true_skills, all_predicted_skills)
    recall = recall_score(all_true_skills, all_predicted_skills)
    f1 = f1_score(all_true_skills, all_predicted_skills)

    accuracy = sum(1 for true, pred in zip(all_true_skills, all_predicted_skills) if true == pred) / len(all_true_skills)
    return accuracy, precision, recall, f1


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

    # Calculate metrics using sklearn
    precision = precision_score(all_true_skills, all_predicted_skills)
    recall = recall_score(all_true_skills, all_predicted_skills)
    f1 = f1_score(all_true_skills, all_predicted_skills)

    accuracy = sum(1 for true, pred in zip(all_true_skills, all_predicted_skills) if true == pred) / len(all_true_skills)
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
    app.run(debug=True)'''







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