import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import logging

logger = logging.getLogger(__name__)


def load_model(config):
    """Charge le modèle Mistral 7B + adaptateur LoRA si disponible."""
    model_name = config["mistral"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        device_map="auto"
    )

    if os.path.exists(config["mistral"]["lora_dir"]):
        model = PeftModel.from_pretrained(model, config["mistral"]["lora_dir"])
        logger.info(f"Adaptateur LoRA chargé depuis {config['mistral']['lora_dir']}")
    else:
        logger.warning(f"Aucun adaptateur LoRA trouvé dans {config['mistral']['lora_dir']}, utilisation du modèle de base.")

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=500,
        device_map="auto"
    )

    return model, tokenizer, pipe


def proofread_french(text: str, charte: str, pipe) -> str:
    """Agent Relecture FR : corrige un texte en français."""
    prompt = (
        "Tu es le rédacteur en chef d'un magazine de luxe alpin philippin.\n"
        f"Charte éditoriale : {charte or 'ton élégant, précis et journalistique, vocabulaire du luxe alpin'}\n"
        f"Corrige le texte suivant en français :\n{text}\n"
        "Texte corrigé :"
    )
    result = pipe(prompt, max_new_tokens=500)[0]['generated_text']
    return result.split("Texte corrigé :")[-1].strip()


def translate_with_mistral(text: str, charte: str, pipe) -> str:
    """Agent Traduction FR->EN via Mistral local (fine-tuné)."""
    prompt = (
        "You are an expert translator specializing in luxury alpine content.\n"
        f"Editorial guidelines: {charte or 'elegant, precise tone for a luxury alpine magazine'}\n"
        f"Translate the following text from French to English:\n{text}\n"
        "English translation:"
    )
    result = pipe(prompt, max_new_tokens=500)[0]['generated_text']
    return result.split("English translation:")[-1].strip()


def translate_with_deepl(text: str, config: dict) -> str:
    """Agent Traduction FR->EN via DeepL API."""
    import deepl
    translator = deepl.Translator(config["deepl"]["api_key"])
    result = translator.translate_text(
        text,
        source_lang=config["deepl"].get("source_lang", "FR"),
        target_lang=config["deepl"].get("target_lang", "EN-GB")
    )
    return result.text


def proofread_english(text: str, charte: str, pipe) -> str:
    """Agent Relecture EN : relit et corrige un texte en anglais."""
    prompt = (
        "You are the editor-in-chief of a luxury alpine magazine.\n"
        f"Editorial guidelines: {charte or 'elegant, precise, journalistic tone for luxury alpine content'}\n"
        f"Proofread the following English text:\n{text}\n"
        "Proofread text:"
    )
    result = pipe(prompt, max_new_tokens=500)[0]['generated_text']
    return result.split("Proofread text:")[-1].strip()


def process_text_with_model(
    text: str,
    action: str,
    charte: str,
    pipe,
    config: dict = None,
    translation_engine: str = "mistral"
):
    """
    Traite le texte selon l'action demandée.

    Actions :
        relecture_fr  -- correction en français
        traduction    -- traduction FR->EN (engine: mistral | deepl | compare)
        relecture_en  -- relecture en anglais
    """
    if action == "relecture_fr":
        return proofread_french(text, charte, pipe)

    elif action == "traduction":
        if translation_engine == "deepl":
            if config is None:
                raise ValueError("config requis pour DeepL")
            return translate_with_deepl(text, config)
        elif translation_engine == "compare":
            if config is None:
                raise ValueError("config requis pour la comparaison")
            return {
                "mistral": translate_with_mistral(text, charte, pipe),
                "deepl": translate_with_deepl(text, config)
            }
        else:  # mistral (défaut)
            return translate_with_mistral(text, charte, pipe)

    elif action == "relecture_en":
        return proofread_english(text, charte, pipe)

    else:
        raise ValueError(
            f"Action inconnue : '{action}'. "
            "Actions valides : relecture_fr, traduction, relecture_en"
        )
