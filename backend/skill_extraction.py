import pdfplumber
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
    "technical": ["django", "python", "sql", "java", "tensorflow"],
    "soft": ["communication", "teamwork", "problem-solving"],
    "managerial": ["leadership", "strategic planning", "budget management", "project management"]
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
            match = process.extractOne(token, skills, score_cutoff=85)
            if match:
                extracted_skills[category].append(match[0])
    return {category: list(set(skills)) for category, skills in extracted_skills.items()}


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