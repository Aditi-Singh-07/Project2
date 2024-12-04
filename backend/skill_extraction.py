import PyPDF2
import spacy
import re

# Load pre-trained model (ensure you've installed spaCy and its model)
nlp = spacy.load("en_core_web_sm")

# Dummy skills list (you can expand this with your own list or use a more advanced method)
SKILLS_LIST = ['Python', 'JavaScript', 'Java', 'SQL', 'Machine Learning', 'Deep Learning', 'Data Science']

def extract_skills(file_path):
    # Read the PDF file
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ''
        for page in reader.pages:
            text += page.extract_text()

    # Preprocess the text (remove extra spaces, newlines)
    text = re.sub(r'\s+', ' ', text)

    # Use spaCy to process the text
    doc = nlp(text)
    
    # Extract skills
    skills = [skill for skill in SKILLS_LIST if skill.lower() in text.lower()]
    experience = 'Extracted experience details'  # Add your logic to extract experience
    projects = 'Extracted project details'  # Add your logic to extract projects

    return skills, experience, projects

