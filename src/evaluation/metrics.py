"""
evaluation/metrics.py
Métricas de evaluación: ROUGE, BLEU y Exact Match.
"""
import math
import re
import unicodedata
from collections import Counter

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
except Exception:  # pragma: no cover - fallback para entornos sin NLTK
    nltk = None
    sentence_bleu = None
    SmoothingFunction = None

try:
    from rouge_score import rouge_scorer
except Exception:  # pragma: no cover - fallback para entornos sin rouge_score
    rouge_scorer = None

from src.utils.logger import get_logger

logger = get_logger(__name__)

if nltk is not None:
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


def _tokenize(text: str) -> list[str]:
    text = normalize_text(text)
    return text.split()


def _rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = [[0] * (len(ref_tokens) + 1) for _ in range(len(pred_tokens) + 1)]
    for i in range(1, len(pred_tokens) + 1):
        for j in range(1, len(ref_tokens) + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                lcs[i][j] = lcs[i - 1][j - 1] + 1
            else:
                lcs[i][j] = max(lcs[i - 1][j], lcs[i][j - 1])

    lcs_len = lcs[len(pred_tokens)][len(ref_tokens)]
    precision = lcs_len / len(pred_tokens)
    recall = lcs_len / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def _rouge_n_f1(prediction: str, reference: str, n: int) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_ngrams = Counter(tuple(pred_tokens[i:i + n]) for i in range(len(pred_tokens) - n + 1))
    ref_ngrams = Counter(tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1))
    if not pred_ngrams or not ref_ngrams:
        return 0.0

    overlap = sum(min(pred_ngrams[g], ref_ngrams[g]) for g in pred_ngrams.keys() & ref_ngrams.keys())
    precision = overlap / sum(pred_ngrams.values())
    recall = overlap / sum(ref_ngrams.values())
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def compute_rouge(prediction: str, reference: str) -> dict:
    """
    Calcula ROUGE-1, ROUGE-2 y ROUGE-L con una implementación ligera.

    Returns:
        Dict con scores F1 para cada métrica.
    """
    return {
        "rouge1": _rouge_n_f1(prediction, reference, 1),
        "rouge2": _rouge_n_f1(prediction, reference, 2),
        "rougeL": _rouge_l_f1(prediction, reference),
    }


def compute_bleu(prediction: str, reference: str) -> float:
    """
    Calcula BLEU a nivel de oración con una implementación ligera.

    Returns:
        Score BLEU entre 0 y 1.
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0

    if sentence_bleu is not None and SmoothingFunction is not None:
        smoother = SmoothingFunction().method1
        score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoother)
        return round(float(score), 4)

    # Fallback simple si NLTK no está disponible
    overlap = sum(min(Counter(pred_tokens)[t], Counter(ref_tokens)[t]) for t in set(pred_tokens) & set(ref_tokens))
    precision = overlap / max(len(pred_tokens), 1)
    return round(float(precision), 4)


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


# ── Métricas específicas de RAG ────────────────────────────────────────────────

def compute_hit_rate(contexts: list[str], reference: str, overlap_threshold: float = 0.3) -> float:
    """
    Hit Rate: ¿alguno de los chunks recuperados cubre la respuesta de referencia?

    Mide si al menos un chunk tiene suficiente overlap de tokens con el ground truth.
    No requiere LLM — es puramente léxico.

    Returns:
        1.0 si hay hit, 0.0 si no.
    """
    if not contexts or not reference:
        return 0.0

    ref_tokens = set(_tokenize(reference))
    if not ref_tokens:
        return 0.0

    for ctx in contexts:
        ctx_tokens = set(_tokenize(ctx))
        overlap = len(ref_tokens & ctx_tokens) / len(ref_tokens)
        if overlap >= overlap_threshold:
            return 1.0

    return 0.0


def compute_context_recall(contexts: list[str], reference: str) -> float:
    """
    Context Recall léxico: fracción de tokens del ground truth cubiertos por los chunks.

    Análogo a ROUGE-1 Recall pero sobre el conjunto unificado de contextos.
    No requiere LLM.
    """
    if not contexts or not reference:
        return 0.0

    ref_tokens = _tokenize(reference)
    if not ref_tokens:
        return 0.0

    ctx_tokens = _tokenize(" ".join(contexts))
    ctx_counter = Counter(ctx_tokens)
    ref_counter = Counter(ref_tokens)

    overlap = sum(min(ref_counter[t], ctx_counter[t]) for t in ref_counter)
    return round(overlap / len(ref_tokens), 4)


def evaluate_rag_single(
    prediction: str,
    reference: str,
    contexts: list[str],
) -> dict:
    """
    Evalúa una muestra RAG con métricas de texto + métricas de recuperación.

    Args:
        prediction: Respuesta generada por el pipeline RAG.
        reference:  Respuesta de referencia (ground truth).
        contexts:   Lista de chunks recuperados por el retriever.

    Returns:
        Dict con ROUGE, BLEU, exact_match, hit_rate y context_recall.
    """
    text_scores = evaluate_single(prediction, reference)
    hit = compute_hit_rate(contexts, reference)
    ctx_recall = compute_context_recall(contexts, reference)

    return {
        **text_scores,
        "hit_rate": hit,
        "context_recall": ctx_recall,
    }


def evaluate_rag_batch(
    predictions: list[str],
    references: list[str],
    contexts_list: list[list[str]],
) -> dict:
    """
    Evalúa un batch completo de ejemplos RAG.

    Args:
        predictions:   Respuestas generadas.
        references:    Ground truths.
        contexts_list: Lista de listas de chunks recuperados (uno por pregunta).

    Returns:
        Dict con métricas promedio (texto + RAG).
    """
    assert len(predictions) == len(references) == len(contexts_list), "Listas de diferente tamaño"

    all_scores = [
        evaluate_rag_single(p, r, c)
        for p, r, c in zip(predictions, references, contexts_list)
    ]

    avg = {}
    for key in all_scores[0].keys():
        avg[key] = round(sum(s[key] for s in all_scores) / len(all_scores), 4)

    logger.info("Métricas RAG promedio del batch:")
    for k, v in avg.items():
        logger.info(f"  {k}: {v}")

    return avg
