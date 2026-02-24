# magazine_ai/app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import torch
import pdfplumber
from docx import Document
import tempfile

# Charger la configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialiser FastAPI
app = FastAPI()

# CORS (pour permettre les requêtes depuis n'importe où en développement)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger le modèle (au démarrage, avec cache pour éviter de recharger à chaque requête)
@app.on_event("startup")
async def load_model():
    global model, tokenizer, pipe

    model_name = config["mistral"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Charger le modèle de base en 4-bit pour économiser la VRAM
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        device_map="auto"
    )

    # Charger le modèle LoRA fine-tuné
    if os.path.exists(config["mistral"]["lora_dir"]):
        model = PeftModel.from_pretrained(model, config["mistral"]["lora_dir"])

    # Créer un pipeline de génération
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=500,
        device_map="auto"
    )

class TextRequest(BaseModel):
    text: str
    action: str  # "relecture" ou "traduction"
    charte: Optional[str] = None

@app.post("/process")
async def process_text(request: TextRequest):
    try:
        if request.action == "relecture":
            prompt = f"""
            Tu es un rédacteur en chef pour un magazine de luxe alpin.
            Corrige ce texte en respectant cette charte : {request.charte or ''}
            Texte à corriger : {request.text}
            Texte corrigé :
            """
        else:  # traduction
            prompt = f"""
            Tu es un traducteur spécialisé dans le luxe alpin.
            Traduis ce texte en anglais en respectant cette charte : {request.charte or ''}
            Texte à traduire : {request.text}
            Traduction :
            """

        result = pipe(prompt)[0]['generated_text']
        if request.action == "relecture":
            corrected_text = result.split("Texte corrigé :")[-1].strip()
        else:
            corrected_text = result.split("Traduction :")[-1].strip()

        return {"status": "success", "result": corrected_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), action: str = "relecture", charte: Optional[str] = None):
    try:
        # Sauvegarder temporairement le fichier
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            temp_path = temp_file.name
            temp_file.write(await file.read())

        # Extraire le texte
        text = extract_text(temp_path)

        # Traiter le texte
        request = TextRequest(text=text, action=action, charte=charte)
        result = await process_text(request)

        # Nettoyer
        os.remove(temp_path)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        raise ValueError("Format de fichier non supporté.")
