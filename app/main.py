
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import yaml
from app.core.inference import load_model, process_text_with_model
from app.utils.extract_text_utils import extract_text
from app.utils.report_utils import generate_report
import tempfile
import logging

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

# Charger le modèle au démarrage
model, tokenizer, pipe = None, None, None

@app.on_event("startup")
async def startup_event():
    global model, tokenizer, pipe
    model, tokenizer, pipe = load_model(config)
    logger.info("Modèle chargé avec succès")

class TextRequest(BaseModel):
    text: str
    action: str  # "relecture" ou "traduction"
    charte: Optional[str] = None

@app.post("/process")
async def process_text(request: TextRequest):
    try:
        result = process_text_with_model(request.text, request.action, request.charte, pipe, tokenizer)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Erreur lors du traitement : {e}")
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
        result = process_text_with_model(text, action, charte, pipe, tokenizer)

        # Générer un rapport
        report_path = generate_report(text, result, config["paths"]["reports"])
        report_url = f"/reports/{os.path.basename(report_path)}"

        # Nettoyer
        os.remove(temp_path)

        return {"status": "success", "result": result, "report_url": report_url}

    except Exception as e:
        logger.error(f"Erreur lors de l'upload : {e}")
        raise HTTPException(status_code=500, detail=str(e))