"""
evaluate_translation.py
-----------------------
Evalue et compare la qualite de traduction FR->EN entre Mistral (local) et DeepL API.

NOTE METHODOLOGIQUE :
    DeepL est utilise comme proxy de reference (pas de traduction humaine disponible).
    Les scores BLEU/ROUGE mesurent la divergence Mistral vs DeepL, pas une verite absolue.
    Un score eleve signifie que Mistral s'aligne sur DeepL.

Usage :
    python scripts/evaluate_translation.py --input data/datasets/synthetic_dataset.jsonl --num-samples 20
    python scripts/evaluate_translation.py --input data/datasets/synthetic_dataset.jsonl --skip-mistral
"""

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _require_sacrebleu():
    try:
        from sacrebleu.metrics import BLEU
        return BLEU
    except ImportError:
        logger.error("sacrebleu n'est pas installe. Lancez : pip install sacrebleu")
        sys.exit(1)


def _require_rouge():
    try:
        from rouge_score import rouge_scorer
        return rouge_scorer
    except ImportError:
        logger.error("rouge-score n'est pas installe. Lancez : pip install rouge-score")
        sys.exit(1)


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        logger.error(f"Configuration introuvable : {config_path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_input_texts(input_path: str, num_samples) -> list:
    path = Path(input_path)
    if not path.exists():
        logger.error(f"Fichier introuvable : {input_path}")
        sys.exit(1)

    texts = []
    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Ligne {lineno} ignoree : {e}")
                    continue
                for field in ("input", "source", "text", "fr"):
                    if field in obj and isinstance(obj[field], str) and obj[field].strip():
                        texts.append(obj[field].strip())
                        break
    elif path.suffix.lower() == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            texts = [l.strip() for l in f if l.strip()]
    else:
        logger.error(f"Format non supporte : {path.suffix}. Utilisez .txt ou .jsonl")
        sys.exit(1)

    if not texts:
        logger.error("Aucun texte trouve.")
        sys.exit(1)

    if num_samples and num_samples < len(texts):
        texts = texts[:num_samples]

    logger.info(f"{len(texts)} texte(s) charge(s).")
    return texts


def load_mistral_pipeline(config: dict):
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from app.core.inference import load_model
    _, _, pipe = load_model(config)
    return pipe


def translate_mistral(text: str, pipe, charte: str = "") -> tuple:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from app.core.inference import translate_with_mistral
    t0 = time.perf_counter()
    translation = translate_with_mistral(text, charte, pipe)
    return translation, round(time.perf_counter() - t0, 3)


def translate_deepl(text: str, config: dict) -> tuple:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from app.core.inference import translate_with_deepl
    t0 = time.perf_counter()
    translation = translate_with_deepl(text, config)
    return translation, round(time.perf_counter() - t0, 3)


def compute_bleu(hypothesis: str, reference: str) -> float:
    BLEU = _require_sacrebleu()
    score = BLEU(effective_order=True).sentence_score(hypothesis, [reference])
    return round(score.score, 2)


def compute_rouge_l(hypothesis: str, reference: str) -> float:
    rouge_scorer_mod = _require_rouge()
    scorer = rouge_scorer_mod.RougeScorer(["rougeL"], use_stemmer=True)
    return round(scorer.score(reference, hypothesis)["rougeL"].fmeasure, 4)


def compute_length_ratio(translation: str, source: str) -> float:
    src_len = len(source.split())
    return round(len(translation.split()) / src_len, 3) if src_len else 0.0


DEEPL_FREE_CHARS = 500_000
DEEPL_PRO_PRICE_PER_CHAR = 25 / 1_000_000

def estimate_deepl_cost(total_chars: int) -> str:
    if total_chars <= DEEPL_FREE_CHARS:
        return "0EUR (Free tier)"
    return f"~{(total_chars - DEEPL_FREE_CHARS) * DEEPL_PRO_PRICE_PER_CHAR:.2f}EUR/mois (Pro)"


CSV_COLUMNS = [
    "text_id", "source_fr", "translation_mistral", "translation_deepl",
    "bleu_mistral", "bleu_deepl", "rouge_mistral", "rouge_deepl",
    "length_ratio_mistral", "length_ratio_deepl", "time_mistral", "time_deepl",
]


def save_report(rows: list, output_path: str):
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Rapport sauvegarde : {out}")


def print_summary(rows, avg_time_m, avg_time_d, deepl_cost):
    n = len(rows)
    valid = [r for r in rows if isinstance(r["bleu_mistral"], float)]
    avg_bleu  = sum(r["bleu_mistral"]  for r in valid) / len(valid) if valid else 0.0
    avg_rouge = sum(r["rouge_mistral"] for r in valid) / len(valid) if valid else 0.0
    avg_lr_m  = sum(r["length_ratio_mistral"] for r in rows if isinstance(r["length_ratio_mistral"], float)) / n
    avg_lr_d  = sum(r["length_ratio_deepl"]   for r in rows if isinstance(r["length_ratio_deepl"],   float)) / n

    sep = "=" * 72
    print(f"\n{sep}")
    print("  RAPPORT D'EVALUATION TRADUCTION  (FR -> EN)")
    print(sep)
    print(f"  Textes evalues       : {n}")
    print(f"  Reference            : DeepL API (proxy methodologique)\n")
    print(f"  {'Methode':<24} {'BLEU':>7} {'ROUGE-L':>9} {'LenRatio':>10} {'Tps moy':>10} {'Cout':>18}")
    print("  " + "-" * 80)
    print(f"  {'Mistral (local)':<24} {avg_bleu:>6.1f}% {avg_rouge:>8.1%} {avg_lr_m:>10.3f} {avg_time_m:>9.2f}s {'0EUR (local)':>18}")
    print(f"  {'DeepL API':<24} {'ref.':>7} {'ref.':>9} {avg_lr_d:>10.3f} {avg_time_d:>9.2f}s {deepl_cost:>18}")
    print(sep)
    print("\n  Interpretation :")
    print("    BLEU     : 0-100  (>30 = acceptable, >50 = bon alignement avec DeepL)")
    print("    ROUGE-L  : 0-1    (>0.5 = bonne couverture de contenu)")
    print("    LenRatio : mots traduits / mots source  (ideal FR->EN : 0.9 - 1.3)")
    print(f"{sep}\n")


def main():
    parser = argparse.ArgumentParser(description="Evalue Mistral vs DeepL (FR->EN)")
    parser.add_argument("--input",  "-i", required=True, help=".txt ou .jsonl (champ 'input')")
    parser.add_argument("--config", "-c", default="config.yaml")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--num-samples", "-n", type=int, default=None)
    parser.add_argument("--charte", default="elegant, precise tone for a luxury alpine magazine")
    parser.add_argument("--skip-mistral", action="store_true", help="Teste DeepL seul (sans GPU)")
    args = parser.parse_args()

    config = load_config(args.config)
    texts  = load_input_texts(args.input, args.num_samples)
    output_path = args.output or f"data/evaluation/report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"

    pipe = None
    if not args.skip_mistral:
        logger.info("Chargement du modele Mistral...")
        pipe = load_mistral_pipeline(config)

    rows = []
    total_deepl_chars = 0

    for idx, source in enumerate(texts, 1):
        logger.info(f"[{idx}/{len(texts)}] {source[:60]}...")

        try:
            trans_deepl, time_deepl = translate_deepl(source, config)
        except Exception as e:
            logger.error(f"DeepL echoue (texte {idx}) : {e}")
            trans_deepl, time_deepl = "", 0.0

        total_deepl_chars += len(source)

        if pipe:
            try:
                trans_mistral, time_mistral = translate_mistral(source, pipe, args.charte)
            except Exception as e:
                logger.error(f"Mistral echoue (texte {idx}) : {e}")
                trans_mistral, time_mistral = "", 0.0
        else:
            trans_mistral, time_mistral = "", 0.0

        bleu_m  = compute_bleu(trans_mistral, trans_deepl)  if (trans_mistral and trans_deepl) else 0.0
        rouge_m = compute_rouge_l(trans_mistral, trans_deepl) if (trans_mistral and trans_deepl) else 0.0

        rows.append({
            "text_id": idx, "source_fr": source,
            "translation_mistral": trans_mistral, "translation_deepl": trans_deepl,
            "bleu_mistral": bleu_m, "bleu_deepl": "ref",
            "rouge_mistral": rouge_m, "rouge_deepl": "ref",
            "length_ratio_mistral": compute_length_ratio(trans_mistral, source) if trans_mistral else 0.0,
            "length_ratio_deepl":   compute_length_ratio(trans_deepl,   source) if trans_deepl   else 0.0,
            "time_mistral": time_mistral, "time_deepl": time_deepl,
        })

    n = len(rows)
    save_report(rows, output_path)
    print_summary(
        rows,
        sum(r["time_mistral"] for r in rows) / n,
        sum(r["time_deepl"]   for r in rows) / n,
        estimate_deepl_cost(total_deepl_chars),
    )


if __name__ == "__main__":
    main()
