import json
from transformers import pipeline

def generate_synthetic_data(num_examples=50):
    """Génère des paires input/output synthétiques pour le fine-tuning."""
    generator = pipeline("text-generation", model="mistralai/Mistral-7B-v0.1")

    # Exemples de phrases "avant" (style non corrigé)
    inputs = [
        "La descente était super raide, mais le chalet au bas était trop cool.",
        "Les skis sont top, mais il fait un froid de canard.",
        "La vue est magnifique, surtout avec ce ciel bleu.",
        "Ce restaurant propose une cuisine locale et des plats typiques.",
        "La neige est poudreuse, parfaite pour les skieurs expérimentés."
    ] * (num_examples // len(inputs) + 1)  # Répète pour atteindre num_examples

    datasets = []
    for input_text in inputs[:num_examples]:
        prompt = f"""
        Corrige ce texte selon la charte suivante :
        - Remplacer "raide" par "abrupte".
        - Remplacer "sympa" ou "cool" par "élégant" ou "chaleureux".
        - Utiliser un ton journalistique et descriptif.
        - Eviter les expressions trop familiaires.
        Utiliser un vocabulaire adapté aux magazines de luxe.

        Texte à corriger : {input_text}
        Texte corrigé :
        """
        result = generator(prompt, max_length=200, num_return_sequences=1)
        output_text = result[0]['generated_text'].split("Texte corrigé :")[-1].strip()

        datasets.append({
            "input": input_text,
            "output": output_text
        })

    # Sauvegarder
    output_dir = "../data/datasets"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "synthetic_dataset.jsonl"), "w", encoding="utf-8") as f:
        for item in datasets:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"{len(datasets)} exemples synthétiques générés dans {output_dir}")

if __name__ == "__main__":
    generate_synthetic_data(num_examples=50)
