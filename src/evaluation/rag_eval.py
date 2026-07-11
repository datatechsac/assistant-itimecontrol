"""
src/evaluation/rag_eval.py
Evalúa el pipeline RAG usando un dataset de preguntas/respuestas.

Métricas calculadas:
  - Texto: ROUGE-1, ROUGE-2, ROUGE-L, BLEU, Exact Match
  - RAG:   Hit Rate@K, Context Recall

Formatos de dataset soportados (JSON / JSONL):
  {"question": "...", "answer": "..."}
  {"question": "...", "ground_truth": "..."}
  {"instruction": "...", "output": "..."}
"""
import argparse
import json
from pathlib import Path
from typing import Any

from src.evaluation.metrics import evaluate_rag_batch
from src.rag.pipeline import RAGPipeline
from src.utils.helpers import load_config, load_json, save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


def normalize_evaluation_item(item: dict[str, Any]) -> dict[str, str] | None:
    """Normaliza un registro del dataset a formato {question, answer}."""
    if not isinstance(item, dict):
        return None

    if "question" in item and "ground_truth" in item:
        return {"question": str(item["question"]), "answer": str(item["ground_truth"])}

    if "question" in item and "answer" in item:
        return {"question": str(item["question"]), "answer": str(item["answer"])}

    if "instruction" in item and "output" in item:
        return {"question": str(item["instruction"]), "answer": str(item["output"])}

    if "input" in item and "output" in item:
        return {"question": str(item["input"]), "answer": str(item["output"])}

    if "prompt" in item and "response" in item:
        return {"question": str(item["prompt"]), "answer": str(item["response"])}

    return None


def load_evaluation_items(
    dataset_path: str | Path, limit: int | None = None
) -> list[dict[str, str]]:
    """Carga ejemplos desde un archivo JSON o JSONL."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo de evaluación no encontrado: {path}")

    if path.suffix.lower() == ".jsonl":
        raw_items = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_items.append(json.loads(line))
    else:
        data = load_json(str(path))
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        if not isinstance(data, list):
            raise ValueError(f"Formato no soportado en {path}: se esperaba lista o JSONL")
        raw_items = data

    normalized = []
    for item in raw_items:
        norm = normalize_evaluation_item(item)
        if norm is None:
            continue
        normalized.append(norm)
        if limit is not None and len(normalized) >= limit:
            break

    if not normalized:
        raise ValueError(f"No se encontraron ejemplos válidos en {path}")

    return normalized


def run_rag_evaluation(
    config: dict,
    dataset_path: str | Path | None = None,
    limit: int | None = None,
    output_path: str | Path | None = None,
) -> dict:
    """Ejecuta evaluación completa del pipeline RAG sobre un dataset."""
    dataset_path = dataset_path or config["evaluation"]["benchmark_file"]
    items = load_evaluation_items(dataset_path, limit=limit)
    logger.info(f"Dataset cargado: {len(items)} ejemplos desde {dataset_path}")

    logger.info("Inicializando pipeline RAG (Claude API)...")
    rag = RAGPipeline(config)

    predictions: list[str] = []
    references: list[str] = []
    contexts_list: list[list[str]] = []
    detailed_results = []

    for i, item in enumerate(items, start=1):
        question = item["question"]
        reference = item["answer"]
        if not question or not reference:
            continue

        logger.info(f"[{i}/{len(items)}] {question[:80]}...")
        result = rag.generate(question)
        prediction = result["answer"]

        predictions.append(prediction)
        references.append(reference)
        contexts_list.append(result["contexts"])

        detailed_results.append({
            "question": question,
            "reference": reference,
            "prediction": prediction,
            "contexts": result["contexts"],
            "sources": result["sources"],
            "num_chunks": result["num_chunks"],
        })

    if not predictions:
        raise ValueError("No se generaron predicciones para evaluar")

    avg_metrics = evaluate_rag_batch(predictions, references, contexts_list)

    logs_dir = Path(config["paths"]["logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = logs_dir / "rag_evaluation_results.json"

    provider = config.get("generation", {}).get("provider", "groq")
    model_name = config.get(provider, {}).get("model", "unknown")

    output = {
        "summary": avg_metrics,
        "total_questions": len(predictions),
        "dataset_path": str(dataset_path),
        "provider": provider,
        "model": model_name,
        "results": detailed_results,
    }
    save_json(output, str(output_path))

    logger.info("\n" + "=" * 45)
    logger.info("RESUMEN DE EVALUACIÓN RAG")
    logger.info("=" * 45)
    for k, v in avg_metrics.items():
        logger.info(f"  {k:18s}: {v:.4f}")
    logger.info("=" * 45)
    logger.info(f"Resultados guardados en: {output_path}")

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluar el pipeline RAG sobre un dataset")
    parser.add_argument("--dataset", default=None, help="Ruta al archivo JSON/JSONL")
    parser.add_argument("--limit", type=int, default=None, help="Máximo de ejemplos a evaluar")
    parser.add_argument("--output", default=None, help="Ruta del JSON de resultados")
    args = parser.parse_args()

    config = load_config()
    run_rag_evaluation(
        config,
        dataset_path=args.dataset,
        limit=args.limit,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
