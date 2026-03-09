import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Profils de magazine
# ---------------------------------------------------------------------------

MAGAZINE_PROFILES = {
    "alpine_luxury": {
        "description_fr": (
            "magazine de luxe alpin philippin dédié aux stations de ski haut de gamme, "
            "à l'art de vivre en altitude et aux expériences exclusives en montagne"
        ),
        "description_en": (
            "a luxury Philippine alpine magazine dedicated to high-end ski resorts, "
            "mountain lifestyle and exclusive altitude experiences"
        ),
        "translation_guidelines": (
            "Preserve the sense of altitude, exclusivity and Alpine refinement. "
            "Use precise ski and mountain vocabulary (piste, off-piste, summit, chalet, "
            "powder snow, apres-ski). Keep the elegant, aspirational tone."
        ),
        "vocabulary_hints_fr": (
            "piste, domaine skiable, chalet, altitude, enneigement, télésiège, "
            "sommet, après-ski, freeride, luxe alpin"
        ),
        "vocabulary_hints_en": (
            "ski run, ski area, chalet, altitude, snowfall, chairlift, "
            "summit, apres-ski, freeride, alpine luxury"
        ),
        "max_new_tokens_proofread": 600,
        "max_new_tokens_translate": 800,
    },
    "decoration": {
        "description_fr": (
            "magazine de décoration intérieure haut de gamme, célébrant l'architecture d'intérieur, "
            "le design contemporain, les matériaux nobles et les tendances lifestyle premium"
        ),
        "description_en": (
            "a high-end interior decoration magazine celebrating interior architecture, "
            "contemporary design, noble materials and premium lifestyle trends"
        ),
        "translation_guidelines": (
            "Preserve the sensory and visual richness of the text. "
            "Use precise design vocabulary (texture, volume, light, palette, bespoke, "
            "atelier, finish, craftsmanship). Maintain an aspirational yet intimate tone."
        ),
        "vocabulary_hints_fr": (
            "mobilier, matière, lumière naturelle, palette chromatique, espace, volume, "
            "savoir-faire, artisan, sur-mesure, design contemporain"
        ),
        "vocabulary_hints_en": (
            "furniture, material, natural light, colour palette, space, volume, "
            "craftsmanship, artisan, bespoke, contemporary design"
        ),
        "max_new_tokens_proofread": 600,
        "max_new_tokens_translate": 800,
    },
}

_DEFAULT_MAGAZINE_TYPE = "alpine_luxury"


def _get_profile(magazine_type: str) -> dict:
    if magazine_type not in MAGAZINE_PROFILES:
        logger.warning(
            f"Type de magazine inconnu '{magazine_type}'. "
            f"Utilisation du profil '{_DEFAULT_MAGAZINE_TYPE}'."
        )
        return MAGAZINE_PROFILES[_DEFAULT_MAGAZINE_TYPE]
    return MAGAZINE_PROFILES[magazine_type]


# ---------------------------------------------------------------------------
# Chargement du modèle
# ---------------------------------------------------------------------------

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
        logger.warning(
            f"Aucun adaptateur LoRA trouvé dans {config['mistral']['lora_dir']}, "
            "utilisation du modèle de base."
        )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=600,
        device_map="auto"
    )

    return model, tokenizer, pipe


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

def proofread_french(text: str, charte: str, pipe, magazine_type: str = "alpine_luxury") -> str:
    """Agent Relecture FR : corrige un texte en français selon le profil du magazine."""
    profile = _get_profile(magazine_type)
    charte_effective = charte or (
        f"ton élégant, précis et journalistique — vocabulaire recommandé : {profile['vocabulary_hints_fr']}"
    )
    prompt = (
        f"Tu es le rédacteur en chef d'un {profile['description_fr']}.\n"
        f"Charte éditoriale : {charte_effective}\n"
        f"Vocabulaire caractéristique : {profile['vocabulary_hints_fr']}\n"
        f"Corrige le texte suivant en français :\n{text}\n"
        "Texte corrigé :"
    )
    result = pipe(prompt, max_new_tokens=profile["max_new_tokens_proofread"])[0]["generated_text"]
    return result.split("Texte corrigé :")[-1].strip()


def translate_with_mistral(text: str, charte: str, pipe, magazine_type: str = "alpine_luxury") -> str:
    """Agent Traduction FR->EN via Mistral local, adapté au profil du magazine."""
    profile = _get_profile(magazine_type)
    charte_effective = charte or (
        f"elegant, precise tone — recommended vocabulary: {profile['vocabulary_hints_en']}"
    )
    prompt = (
        f"You are an expert translator specializing in content for {profile['description_en']}.\n"
        f"Editorial guidelines: {charte_effective}\n"
        f"Translation guidelines: {profile['translation_guidelines']}\n"
        f"Key vocabulary to favour: {profile['vocabulary_hints_en']}\n"
        f"Translate the following text from French to English:\n{text}\n"
        "English translation:"
    )
    result = pipe(prompt, max_new_tokens=profile["max_new_tokens_translate"])[0]["generated_text"]
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


def proofread_english(text: str, charte: str, pipe, magazine_type: str = "alpine_luxury") -> str:
    """Agent Relecture EN : relit et corrige un texte en anglais selon le profil du magazine."""
    profile = _get_profile(magazine_type)
    charte_effective = charte or (
        f"elegant, precise, journalistic tone — recommended vocabulary: {profile['vocabulary_hints_en']}"
    )
    prompt = (
        f"You are the editor-in-chief of {profile['description_en']}.\n"
        f"Editorial guidelines: {charte_effective}\n"
        f"Key vocabulary to favour: {profile['vocabulary_hints_en']}\n"
        f"Proofread the following English text:\n{text}\n"
        "Proofread text:"
    )
    result = pipe(prompt, max_new_tokens=profile["max_new_tokens_proofread"])[0]["generated_text"]
    return result.split("Proofread text:")[-1].strip()


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def process_text_with_model(
    text: str,
    action: str,
    charte: str,
    pipe,
    config: dict = None,
    translation_engine: str = "mistral",
    magazine_type: str = "alpine_luxury"
):
    """
    Actions : relecture_fr | traduction | relecture_en
    Engines  : mistral | deepl | compare
    Types    : alpine_luxury | decoration | gastronomie | voyage
    """
    if action == "relecture_fr":
        return proofread_french(text, charte, pipe, magazine_type=magazine_type)

    elif action == "traduction":
        if translation_engine == "deepl":
            if config is None:
                raise ValueError("config requis pour DeepL")
            return translate_with_deepl(text, config)
        elif translation_engine == "compare":
            if config is None:
                raise ValueError("config requis pour la comparaison")
            return {
                "mistral": translate_with_mistral(text, charte, pipe, magazine_type=magazine_type),
                "deepl": translate_with_deepl(text, config),
            }
        else:
            return translate_with_mistral(text, charte, pipe, magazine_type=magazine_type)

    elif action == "relecture_en":
        return proofread_english(text, charte, pipe, magazine_type=magazine_type)

    else:
        raise ValueError(
            f"Action inconnue : '{action}'. "
            "Actions valides : relecture_fr, traduction, relecture_en"
        )
