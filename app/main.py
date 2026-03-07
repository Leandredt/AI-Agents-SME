from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Union
import os
import yaml
from app.core.inference import load_model, process_text_with_model
from app.utils.extract_text_utils import extract_text
from app.utils.report_utils import generate_report
import tempfile
import logging

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

app = FastAPI(title="Magazine Alpin — Agents IA")

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


class TextRequest(BaseModel):
    text: str
    action: str              # "relecture_fr" | "traduction" | "relecture_en"
    charte: Optional[str] = None
    translation_engine: str = "mistral"  # "mistral" | "deepl" | "compare"


@app.post("/process")
async def process_text(request: TextRequest):
    try:
        result = process_text_with_model(
            request.text,
            request.action,
            request.charte,
            pipe,
            config=config,
            translation_engine=request.translation_engine
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Erreur lors du traitement : {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    action: str = "relecture_fr",
    charte: Optional[str] = None,
    translation_engine: str = "mistral"
):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            tmp_path = tmp.name
            tmp.write(await file.read())

        text = extract_text(tmp_path)
        result = process_text_with_model(
            text, action, charte, pipe,
            config=config,
            translation_engine=translation_engine
        )

        result_str = result if isinstance(result, str) else str(result)
        report_path = generate_report(text, result_str, config["paths"]["reports"])
        report_url = f"/reports/{os.path.basename(report_path)}"

        os.remove(tmp_path)

        return {"status": "success", "result": result, "report_url": report_url}

    except Exception as e:
        logger.error(f"Erreur lors de l'upload : {e}")
        raise HTTPException(status_code=500, detail=str(e))
