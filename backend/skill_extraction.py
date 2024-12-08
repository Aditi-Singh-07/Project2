import json
from sklearn.metrics import precision_score, recall_score, f1_score
import pdfplumber
import spacy
from rapidfuzz import process
import logging

# Initialize logging for debugging and tracking issues
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the SpaCy English NLP model
nlp = spacy.load("en_core_web_sm")

# Predefined skill ontology for categorization
SKILL_ONTOLOGY = {
    "technical": ["django", "python", "sql", "java", "tensorflow"],
    "soft": ["communication", "teamwork", "problem-solving"],
    "managerial": ["leadership", "strategic planning", "management"]
}

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a given PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text content from the PDF.
    """
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        if not text.strip():
            logger.warning("No text found in the PDF.")
        return text
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        return ""

def preprocess_text(text):
    """
    Preprocess text by tokenizing, lemmatizing, and removing stopwords and punctuation.

    Args:
        text (str): Raw text data.

    Returns:
        list: Processed tokens from the text.
    """
    doc = nlp(text.lower())
    return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

def extract_skills_with_fuzzy(text):
    """
    Extract skills from text using fuzzy matching.

    Args:
        text (str): Text data from which skills need to be extracted.

    Returns:
        dict: Skills categorized as technical, soft, and managerial.
    """
    tokens = preprocess_text(text)  # Preprocess the text
    extracted_skills = {"technical": [], "soft": [], "managerial": []}

    for token in tokens:
        for category, skills in SKILL_ONTOLOGY.items():
            # Use RapidFuzz for fuzzy matching with a cutoff score of 90
            match = process.extractOne(token, skills, score_cutoff=90)
            if match:
                extracted_skills[category].append(match[0])

    # Remove duplicates within each category
    return {category: list(set(skills)) for category, skills in extracted_skills.items()}

def calculate_accuracy(test_folder, expected_skills_file):
    """
    Calculate the precision, recall, and F1 score of the skill extraction process.

    Args:
        test_folder (str): Path to the folder containing test PDF files.
        expected_skills_file (str): Path to the JSON file containing ground truth skills.

    Returns:
        dict: Metrics including precision, recall, and F1 score.
    """
    try:
        # Load expected skills from JSON file
        with open(expected_skills_file, 'r') as f:
            expected_skills = json.load(f)
        
        # Initialize lists for binary classification metrics
        y_true = []
        y_pred = []

        # Process each PDF file in the test folder
        for pdf_file, true_skills in expected_skills.items():
            pdf_path = os.path.join(test_folder, pdf_file)
            if not os.path.exists(pdf_path):
                logger.warning(f"{pdf_file} not found in test folder.")
                continue
            
            # Extract skills from the PDF
            extracted_skills = extract_skills_with_fuzzy(extract_text_from_pdf(pdf_path))
            predicted_skills = [skill for category in extracted_skills.values() for skill in category]

            # Convert to binary labels for metrics calculation
            all_skills = set(true_skills + predicted_skills)
            y_true.extend([1 if skill in true_skills else 0 for skill in all_skills])
            y_pred.extend([1 if skill in predicted_skills else 0 for skill in all_skills])
        
        # Calculate precision, recall, and F1 score using sklearn
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        return {
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1": round(f1, 2)
        }
    except Exception as e:
        logger.error(f"Error calculating accuracy: {e}")
        return {"precision": 0, "recall": 0, "f1": 0}








'''import json
from sklearn.metrics import precision_score, recall_score, f1_score
import pdfplumber
import spacy
from rapidfuzz import process
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Skill Ontology
SKILL_ONTOLOGY = {
    "technical": ["django", "python", "sql", "java", "tensorflow"],
    "soft": ["communication", "teamwork", "problem-solving"],
    "managerial": ["leadership", "strategic planning", "management"]
}

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF."""
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        if not text.strip():
            logger.warning("No text found in the PDF.")
        return text
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        return ""

def preprocess_text(text):
    """Preprocess text by tokenizing, lemmatizing, and removing stopwords."""
    doc = nlp(text.lower())
    return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

def extract_skills_with_fuzzy(text):
    """Extract skills using fuzzy matching."""
    tokens = preprocess_text(text)
    extracted_skills = {"technical": [], "soft": [], "managerial": []}
    for token in tokens:
        for category, skills in SKILL_ONTOLOGY.items():
            match = process.extractOne(token, skills, score_cutoff=90)
            if match:
                extracted_skills[category].append(match[0])
    return {category: list(set(skills)) for category, skills in extracted_skills.items()}

def calculate_accuracy(test_folder, expected_skills_file):
    """Calculate accuracy of the skill extraction."""
    try:
        # Load expected skills
        with open(expected_skills_file, 'r') as f:
            expected_skills = json.load(f)
        
        # Initialize metrics
        y_true = []
        y_pred = []

        # Test each PDF
        for pdf_file, true_skills in expected_skills.items():
            pdf_path = os.path.join(test_folder, pdf_file)
            if not os.path.exists(pdf_path):
                logger.warning(f"{pdf_file} not found in test folder.")
                continue
            
            # Extract skills from PDF
            extracted_skills = extract_skills_with_fuzzy(extract_text_from_pdf(pdf_path))
            predicted_skills = [skill for category in extracted_skills.values() for skill in category]

            # Convert to binary labels for comparison
            all_skills = set(true_skills + predicted_skills)
            y_true.extend([1 if skill in true_skills else 0 for skill in all_skills])
            y_pred.extend([1 if skill in predicted_skills else 0 for skill in all_skills])
        
        # Calculate precision, recall, and F1
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        return {
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1": round(f1, 2)
        }
    except Exception as e:
        logger.error(f"Error calculating accuracy: {e}")
        return {"precision": 0, "recall": 0, "f1": 0}
'''

'''import pdfplumber
import spacy
from rapidfuzz import process
from transformers import T5ForConditionalGeneration, T5Tokenizer
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Load T5 model and tokenizer for MCQ generation
model_name = "t5-small"  # Lightweight, free model
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Skill Ontology
SKILL_ONTOLOGY = {
    "technical": ["django", "python", "sql", "java", "tensorflow","javascript","html","mysql","css","front end development","backend development"],
    "soft": ["communication", "teamwork", "problem-solving"],
    "managerial": ["leadership", "strategic planning", "management"]
}

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF."""
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        if not text.strip():
            logger.warning("No text found in the PDF.")
        return text
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        return ""

def preprocess_text(text):
    """Preprocess text by tokenizing, lemmatizing, and removing stopwords."""
    doc = nlp(text.lower())
    return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

def extract_skills_with_fuzzy(text):
    """Extract skills using fuzzy matching."""
    tokens = preprocess_text(text)
    extracted_skills = {"technical": [], "soft": [], "managerial": []}
    for token in tokens:
        for category, skills in SKILL_ONTOLOGY.items():
            match = process.extractOne(token, skills, score_cutoff=90)
            if match:
                extracted_skills[category].append(match[0])
    return {category: list(set(skills)) for category, skills in extracted_skills.items()}'''


'''import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


def extract_skills(resume_path):
    # Simulating skill extraction for now
    # Replace this logic with actual preprocessing & extraction techniques
    with open(resume_path, 'rb') as f:
        content = f.read()

    # Placeholder processing
    doc = nlp(content.decode('latin1', errors='ignore'))  # Use appropriate encoding
    skills = ["Python", "JavaScript", "Machine Learning"]
    experience = "3 years of software development"
    projects = ["AI Chatbot", "E-commerce platform"]

    return skills, experience, projects
'''