from PyPDF2 import PdfReader

def preprocess_resume(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text
