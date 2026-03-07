import json
import os
import argparse
import pdfplumber
from transformers import pipeline


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
                for paragraph in page_text.split("\n\n"):
                    paragraph = paragraph.strip().replace("\n", " ")
                    if len(paragraph) >= min_length and not _is_noise(paragraph):
                        texts.append(paragraph)

    print(f"{len(texts)} paragraphes extraits de {len(pdf_files)} PDF(s).")
    return texts


def _is_noise(text: str) -> bool:
    """Détecte les éléments non pertinents : numéros de page, en-têtes, URLs..."""
    stripped = text.strip()
    if len(stripped.split()) < 12:
        return True
    if stripped.isdigit():
        return True
    if stripped.startswith("http"):
        return True
    # Ligne qui ressemble à un en-tête (tout en majuscules)
    if stripped.isupper():
        return True
    return False


def _default_inputs() -> list:
    """Exemples de secours si aucun PDF n'est fourni."""
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

    # Utilise les vrais textes extraits des PDFs si disponibles
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
    parser.add_argument(
        "--pdf-dir", type=str, default=None,
        help="Répertoire contenant les PDFs des anciens magazines (optionnel)"
    )
    parser.add_argument(
        "--num-examples", type=int, default=50,
        help="Nombre d'exemples à générer (défaut : 50)"
    )
    args = parser.parse_args()
    generate_synthetic_data(num_examples=args.num_examples, pdf_dir=args.pdf_dir)
