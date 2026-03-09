from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
import os
import yaml
from app.core.inference import load_model, process_text_with_model
from app.utils.extract_text_utils import extract_text
import tempfile
import logging
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

app = FastAPI(
    title="Magazine Alpin — Agents IA",
    description=(
        "API de relecture et de traduction pour un magazine de luxe alpin. "
        "Propulsée par un modèle Mistral fine-tuné avec LoRA."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model, tokenizer, pipe = None, None, None

@app.on_event("startup")
async def startup_event():
    global model, tokenizer, pipe
    model, tokenizer, pipe = load_model(config)
    logger.info("Modèle chargé avec succès")


MAGAZINE_TYPES = Literal["alpine_luxury", "decoration", "gastronomie", "voyage"]

class TextRequest(BaseModel):
    text: str = Field(..., description="Texte à traiter (maximum 5 000 caractères).")
    action: str = Field(..., description="Action : 'relecture_fr', 'traduction', ou 'relecture_en'.")
    charte: Optional[str] = Field(None, description="Charte éditoriale optionnelle.")
    translation_engine: str = Field("mistral", description="Moteur : 'mistral', 'deepl', ou 'compare'.")
    magazine_type: MAGAZINE_TYPES = Field("alpine_luxury", description="Type de magazine ciblé.")

    @field_validator("text")
    @classmethod
    def validate_text_length(cls, v: str) -> str:
        if len(v) > 5000:
            raise ValueError(
                f"Le texte dépasse la limite de 5 000 caractères ({len(v)} fournis). "
                "Veuillez le raccourcir."
            )
        return v


@app.get("/health", tags=["Système"], summary="État de santé de l'API")
async def health():
    model_loaded = pipe is not None
    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "version": app.version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/model/status", tags=["Système"], summary="Statut détaillé du modèle")
async def model_status():
    return {
        "model_name": config.get("mistral", {}).get("model_name", "inconnu"),
        "lora_loaded": os.path.exists(config.get("mistral", {}).get("lora_dir", "")),
        "device": "auto",
    }


@app.post("/process", tags=["Traitement"], summary="Traiter un texte brut")
async def process_text(request: TextRequest):
    try:
        result = process_text_with_model(
            request.text,
            request.action,
            request.charte,
            pipe,
            config=config,
            translation_engine=request.translation_engine,
            magazine_type=request.magazine_type,
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Erreur lors du traitement : {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload", tags=["Traitement"], summary="Traiter un fichier uploadé")
async def upload_file(
    file: UploadFile = File(...),
    action: str = "relecture_fr",
    charte: Optional[str] = None,
    translation_engine: str = "mistral",
    magazine_type: MAGAZINE_TYPES = "alpine_luxury",
):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            tmp_path = tmp.name
            tmp.write(await file.read())

        text = extract_text(tmp_path)
        os.remove(tmp_path)

        if len(text) > 5000:
            raise HTTPException(
                status_code=422,
                detail=f"Le texte extrait dépasse 5 000 caractères ({len(text)}). Réduisez la taille du document.",
            )

        result = process_text_with_model(
            text, action, charte, pipe,
            config=config,
            translation_engine=translation_engine,
            magazine_type=magazine_type,
        )
        return {"status": "success", "result": result}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de l'upload : {e}")
        raise HTTPException(status_code=500, detail=str(e))
