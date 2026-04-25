"""
evaluation/benchmark.py
Evalúa el modelo RAG+fine-tuned contra el conjunto de preguntas de benchmark.

Uso:
    python src/evaluation/benchmark.py
"""
import json
from pathlib import Path

from src.evaluation.metrics import evaluate_batch
from src.rag.pipeline import RAGPipeline
from src.utils.helpers import load_config, load_json, save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_benchmark(config: dict) -> dict:
    """
    Ejecuta el benchmark completo y guarda los resultados.

    Returns:
        Dict con métricas promedio y resultados detallados.
    """
    benchmark_file = config["evaluation"]["benchmark_file"]

    if not Path(benchmark_file).exists():
        logger.error(f"Archivo de benchmark no encontrado: {benchmark_file}")
        logger.error("Ejecuta primero: python src/preprocessing/dataset_builder.py")
        return {}

    benchmark = load_json(benchmark_file)
    logger.info(f"Benchmark cargado: {len(benchmark)} preguntas")

    logger.info("Inicializando RAG Pipeline...")
    rag = RAGPipeline(config)

    predictions = []
    references  = []
    detailed_results = []

    logger.info("Generando respuestas del modelo...")
    for i, item in enumerate(benchmark, start=1):
        question  = item.get("question", "")
        reference = item.get("answer", "")

        if not question or not reference:
            continue

        logger.info(f"  [{i}/{len(benchmark)}] {question[:70]}...")

        result = rag.generate(question)
        prediction = result["answer"]

        predictions.append(prediction)
        references.append(reference)

        detailed_results.append({
            "question":   question,
            "reference":  reference,
            "prediction": prediction,
            "sources":    result["sources"],
            "num_chunks": result["num_chunks"],
        })

    if not predictions:
        logger.error("No se generaron predicciones.")
        return {}

    # Calcular métricas
    avg_metrics = evaluate_batch(predictions, references)

    # Guardar resultados
    output = {
        "summary": avg_metrics,
        "total_questions": len(predictions),
        "results": detailed_results,
    }

    results_dir = Path(config["paths"]["logs_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "benchmark_results.json"
    save_json(output, str(results_path))

    logger.info(f"\nResultados guardados: {results_path}")
    logger.info("\n" + "=" * 40)
    logger.info("RESUMEN DE EVALUACIÓN")
    logger.info("=" * 40)
    for k, v in avg_metrics.items():
        logger.info(f"  {k:15s}: {v:.4f}")
    logger.info("=" * 40)

    return output


def main():
    config = load_config()

    logger.info("=" * 60)
    logger.info("EVALUACIÓN: iTimeControl Assistant")
    logger.info("=" * 60)

    run_benchmark(config)


if __name__ == "__main__":
    main()
