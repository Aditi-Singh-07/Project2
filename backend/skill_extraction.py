import spacy

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
