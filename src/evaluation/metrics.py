"""
evaluation/metrics.py
Métricas de evaluación: ROUGE, BLEU y Exact Match.
"""
import re
import unicodedata

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Descargar recursos NLTK si no están presentes
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


def normalize_text(text: str) -> str:
    """Normaliza texto para comparación: minúsculas, sin acentos, sin puntuación."""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_rouge(prediction: str, reference: str) -> dict:
    """
    Calcula ROUGE-1, ROUGE-2 y ROUGE-L.

    Returns:
        Dict con scores F1 para cada métrica.
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False
    )
    scores = scorer.score(reference, prediction)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def compute_bleu(prediction: str, reference: str) -> float:
    """
    Calcula BLEU a nivel de oración con suavizado.

    Returns:
        Score BLEU entre 0 y 1.
    """
    pred_tokens = normalize_text(prediction).split()
    ref_tokens  = normalize_text(reference).split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    smoother = SmoothingFunction().method1
    score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoother)
    return round(score, 4)


def compute_exact_match(prediction: str, reference: str) -> bool:
    """Verifica si la predicción es exactamente igual a la referencia (normalizado)."""
    return normalize_text(prediction) == normalize_text(reference)


def evaluate_single(prediction: str, reference: str) -> dict:
    """
    Evalúa una sola predicción contra su referencia.

    Returns:
        Dict con todas las métricas.
    """
    rouge = compute_rouge(prediction, reference)
    bleu  = compute_bleu(prediction, reference)
    em    = compute_exact_match(prediction, reference)

    return {
        **rouge,
        "bleu": bleu,
        "exact_match": int(em),
    }


def evaluate_batch(predictions: list[str], references: list[str]) -> dict:
    """
    Evalúa un batch de predicciones.

    Args:
        predictions: Lista de respuestas generadas por el modelo.
        references:  Lista de respuestas de referencia.

    Returns:
        Dict con métricas promedio.
    """
    assert len(predictions) == len(references), "Listas de diferente tamaño"

    all_scores = [evaluate_single(p, r) for p, r in zip(predictions, references)]

    avg = {}
    for key in all_scores[0].keys():
        avg[key] = round(sum(s[key] for s in all_scores) / len(all_scores), 4)

    logger.info("Métricas promedio del batch:")
    for k, v in avg.items():
        logger.info(f"  {k}: {v}")

    return avg
