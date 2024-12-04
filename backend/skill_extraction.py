import re
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def extract_skills_from_text(resume_text):
    predefined_skills = ["django", "python", "sql", "java", "communication", "teamwork", "leadership"]

    tokens = word_tokenize(resume_text.lower())
    tokens = [re.sub(r'\W+', '', token) for token in tokens if token.isalpha()]

    technical_skills = [skill for skill in predefined_skills if skill in tokens]
    soft_skills = [skill for skill in ["communication", "teamwork", "leadership"] if skill in tokens]

    return {
        "technical": technical_skills,
        "soft": soft_skills
    }
