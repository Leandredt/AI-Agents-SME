import pdfplumber
from docx import Document
import os

def extract_text(file_path: str) -> str:
    """Extrait le texte d'un fichier PDF, Docx ou Txt."""
    if file_path.endswith('.pdf'):
        with pdfplumber.open(file_path) as pdf:
            return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    elif file_path.endswith('.docx'):
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise ValueError("Format de fichier non supporté : " + file_path)
