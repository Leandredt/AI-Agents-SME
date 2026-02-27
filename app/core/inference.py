from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import torch
import logging

logger = logging.getLogger(__name__)

def load_model(config):
    """Charge le modèle Mistral + LoRA."""
    model_name = config["mistral"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Charge le modèle de base en 4-bit
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        device_map="auto"
    )

    # Charge le modèle LoRA si disponible
    if os.path.exists(config["mistral"]["lora_dir"]):
        model = PeftModel.from_pretrained(model, config["mistral"]["lora_dir"])
        logger.info(f"Modèle LoRA chargé depuis {config['mistral']['lora_dir']}")
    else:
        logger.warning(f"Aucun modèle LoRA trouvé dans {config['mistral']['lora_dir']}")

    # Crée un pipeline de génération
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=500,
        device_map="auto"
    )

    return model, tokenizer, pipe

def process_text_with_model(text, action, charte, pipe, tokenizer):
    """Traite le texte avec le modèle (relecture ou traduction)."""
    if action == "relecture":
        prompt = f"""
        Tu es un rédacteur en chef pour un magazine de luxe alpin.
        Corrige ce texte en respectant cette charte : {charte or ''}
        Texte à corriger : {text}
        Texte corrigé :
        """
    else:  # traduction
        prompt = f"""
        Tu es un traducteur spécialisé dans le luxe alpin.
        Traduis ce texte en anglais en respectant cette charte : {charte or ''}
        Texte à traduire : {text}
        Traduction :
        """

    result = pipe(prompt, max_new_tokens=500)[0]['generated_text']

    if action == "relecture":
        return result.split("Texte corrigé :")[-1].strip()
    else:
        return result.split("Traduction :")[-1].strip()
