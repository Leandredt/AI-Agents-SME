import json
import os
import re
import argparse
import pdfplumber
from transformers import pipeline


# ---------------------------------------------------------------------------
# Patterns de bruit compilés une seule fois
# ---------------------------------------------------------------------------

_RE_PRICE = re.compile(
    r"(\d+\s*€|€\s*\d+|CHF\s*\d+|\d+\s*CHF|dès\s+\d+|offre\s+spéciale"
    r"|prix\s*:?\s*\d+|promo(?:tion)?|tarif\s*:?\s*\d+)",
    re.IGNORECASE,
)
_RE_REPEATED_CHARS = re.compile(r"(.)\1{4,}|[-_=*•●■▪◆~]{3,}|[.]{3,}")
_RE_URL = re.compile(r"https?://\S+|www\.\S+\.\S+|\S+@\S+\.\S+")
_RE_PHONE = re.compile(
    r"(\+33|\+41|\+32|\+49|\+39)[\s.\-]?(\d[\s.\-]?){8,}"
    r"|0\d([\s.\-]?\d{2}){4}"
    r"|\b\d{2,3}[\s.\-]\d{3}[\s.\-]\d{2}[\s.\-]\d{2}\b"
)
_RE_ADDRESS = re.compile(
    r"\b\d{1,4}[,\s]+(?:rue|avenue|av\.|boulevard|bd\.?|chemin|impasse|allée|place|route)\b"
    r"|\b(?:F-|CH-)?\d{4,5}\b\s+[A-ZÀÂÉÈÊËÎÏÔÙÛÜ][a-zàâéèêëîïôùûü]",
    re.IGNORECASE,
)
_RE_PAGE_REF = re.compile(r"\b(?:p\.?\s*\d+|page\s+\d+|n°\s*\d+|pp\.\s*\d+)\b", re.IGNORECASE)
_RE_SOCIAL = re.compile(
    r"@[\w.]{2,}|#[\w]{2,}"
    r"|\b(?:instagram|facebook|twitter|linkedin|tiktok|youtube|snapchat)\b",
    re.IGNORECASE,
)
_RE_PUNCT_HEAVY = re.compile(r"[^\w\s]")
_RE_NEWLINE_BEFORE_UPPER = re.compile(r"\n(?=\s*[A-ZÀÂÉÈÊËÎÏÔÙÛÜ])")


# ---------------------------------------------------------------------------
# Détection de bruit
# ---------------------------------------------------------------------------

def _is_noise(text: str) -> bool:
    """
    Détecte les éléments non pertinents extraits de PDFs de magazines :
    pubs, prix, caractères répétés, URLs, téléphones, adresses, réseaux sociaux,
    références de pages, ponctuation excessive, successions de noms propres.
    """
    stripped = text.strip()

    if not stripped:
        return True

    words = stripped.split()
    num_words = len(words)

    if num_words < 12:
        return True
    if stripped.isdigit():
        return True
    if stripped.isupper():
        return True

    # Publicité : 2+ occurrences de prix/promo dans un même bloc
    if len(_RE_PRICE.findall(stripped)) >= 2:
        return True
    if _RE_REPEATED_CHARS.search(stripped):
        return True
    if _RE_URL.search(stripped):
        return True
    if _RE_PHONE.search(stripped):
        return True
    if _RE_ADDRESS.search(stripped):
        return True
    if _RE_PAGE_REF.search(stripped):
        return True
    if _RE_SOCIAL.search(stripped):
        return True

    # Texte majoritairement ponctuation (> 40 %)
    non_space = [c for c in stripped if c != " "]
    if non_space and len(_RE_PUNCT_HEAVY.findall(stripped)) / len(non_space) > 0.40:
        return True

    # Majorité de tokens capitalisés (> 50 %) → liste de noms / menu
    capitalized = [w for w in words if len(w) > 1 and w[0].isupper() and w[1:].islower()]
    if num_words > 0 and len(capitalized) / num_words > 0.50:
        return True

    return False


def _has_too_many_allcaps_tokens(paragraph: str, threshold: float = 0.40) -> bool:
    """True si > threshold des tokens sont entièrement en majuscules (longueur > 1)."""
    tokens = paragraph.split()
    if not tokens:
        return False
    allcaps = [t for t in tokens if len(t) > 1 and t.isupper()]
    return len(allcaps) / len(tokens) > threshold


def _split_into_paragraphs(page_text: str) -> list:
    """Double découpage : \n\n puis \n suivi d'une majuscule."""
    paragraphs = []
    for block in page_text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        for sub in _RE_NEWLINE_BEFORE_UPPER.split(block):
            cleaned = re.sub(r" {2,}", " ", sub.strip().replace("\n", " "))
            if cleaned:
                paragraphs.append(cleaned)
    return paragraphs


# ---------------------------------------------------------------------------
# Extraction PDF
# ---------------------------------------------------------------------------

def extract_texts_from_pdfs(pdf_dir: str, min_length: int = 120) -> list:
    """Extrait et filtre les paragraphes pertinents des PDFs du magazine."""
    texts = []

    if not os.path.exists(pdf_dir):
        print(f"Répertoire '{pdf_dir}' introuvable.")
        return texts

    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"Aucun PDF trouvé dans '{pdf_dir}'.")
        return texts

    for filename in pdf_files:
        pdf_path = os.path.join(pdf_dir, filename)
        print(f"Extraction : {filename}")
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if not page_text:
                    continue
                for paragraph in _split_into_paragraphs(page_text):
                    if len(paragraph) < min_length:
                        continue
                    if _has_too_many_allcaps_tokens(paragraph):
                        continue
                    if _is_noise(paragraph):
                        continue
                    texts.append(paragraph)

    print(f"{len(texts)} paragraphes extraits de {len(pdf_files)} PDF(s).")
    return texts


# ---------------------------------------------------------------------------
# Génération de données synthétiques
# ---------------------------------------------------------------------------

def _default_inputs() -> list:
    return [
        "La descente était super raide, mais le chalet au bas était trop cool.",
        "Les skis sont top, mais il fait un froid de canard.",
        "La vue est magnifique, surtout avec ce ciel bleu.",
        "Ce restaurant propose une cuisine locale et des plats typiques de la région.",
        "La neige est poudreuse, parfaite pour les skieurs expérimentés.",
    ]


def generate_synthetic_data(num_examples: int = 50, pdf_dir: str = None):
    """Génère des paires input/output synthétiques pour le fine-tuning."""
    generator = pipeline("text-generation", model="mistralai/Mistral-7B-v0.1")

    if pdf_dir:
        inputs = extract_texts_from_pdfs(pdf_dir)
        if not inputs:
            print("Aucun texte extrait. Utilisation des exemples par défaut.")
            inputs = _default_inputs()
    else:
        print("Aucun répertoire PDF fourni. Utilisation des exemples par défaut.")
        inputs = _default_inputs()

    inputs = inputs[:num_examples]

    charte_prompt = (
        "- Remplacer les expressions familières par un registre élégant et journalistique.\n"
        "- Vocabulaire adapté aux magazines de luxe alpin.\n"
        "- Ton descriptif, précis, sans superlatifs excessifs.\n"
        "- Conserver le sens et les informations factuelles."
    )

    datasets = []
    for i, input_text in enumerate(inputs):
        prompt = (
            f"Corrige ce texte selon la charte suivante :\n{charte_prompt}\n\n"
            f"Texte à corriger : {input_text}\n"
            "Texte corrigé :"
        )
        result = generator(prompt, max_length=512, num_return_sequences=1)
        output_text = result[0]['generated_text'].split("Texte corrigé :")[-1].strip()
        datasets.append({"input": input_text, "output": output_text})
        print(f"[{i+1}/{len(inputs)}] Exemple généré.")

    output_dir = os.path.join(os.path.dirname(__file__), "../data/datasets")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "synthetic_dataset.jsonl")

    with open(output_path, "w", encoding="utf-8") as f:
        for item in datasets:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n{len(datasets)} exemples sauvegardés dans {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génère un dataset synthétique pour le fine-tuning.")
    parser.add_argument("--pdf-dir", type=str, default=None,
                        help="Répertoire contenant les PDFs des anciens magazines")
    parser.add_argument("--num-examples", type=int, default=50,
                        help="Nombre d'exemples à générer (défaut : 50)")
    args = parser.parse_args()
    generate_synthetic_data(num_examples=args.num_examples, pdf_dir=args.pdf_dir)
