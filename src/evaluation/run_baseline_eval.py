"""
evaluation/run_baseline_eval.py
Baseline: el LLM responde SOLO con su conocimiento general, sin recuperación (sin RAG).
Sirve como punto de comparación contra el pipeline RAG completo.

Uso:
    python -m src.evaluation.run_baseline_eval
"""
import argparse
from pathlib import Path

from src.evaluation.llm_judge import judge_batch
from src.evaluation.metrics import evaluate_batch
from src.evaluation.rag_eval import load_evaluation_items
from src.rag.pipeline import RAGPipeline
from src.utils.helpers import load_config, save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)

BASELINE_SYSTEM_PROMPT = (
    "Eres un asistente conversacional en español. "
    "Responde la pregunta del usuario de la forma más precisa posible usando "
    "únicamente tu conocimiento general. Si no conoces la respuesta, dilo claramente."
)


def call_llm_no_context(rag: RAGPipeline, question: str) -> str:
    """Llama al LLM configurado SIN contexto recuperado (baseline puro)."""
    if rag.provider == "groq":
        resp = rag.client.chat.completions.create(
            model=rag._model,
            max_tokens=rag._max_tokens,
            temperature=rag._temperature,
            messages=[
                {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
        )
        return resp.choices[0].message.content.strip()
    raise NotImplementedError(f"Baseline no implementado para proveedor: {rag.provider}")


def run_baseline_evaluation(
    config: dict,
    dataset_path: str,
    output_path: str | None = None,
    run_judge: bool = True,
) -> dict:
    items = load_evaluation_items(dataset_path)
    logger.info(f"Dataset cargado: {len(items)} ejemplos desde {dataset_path}")

    logger.info("Inicializando cliente LLM (sin retriever, baseline puro)...")
    rag = RAGPipeline(config)

    predictions, references, judge_items = [], [], []
    for i, item in enumerate(items, start=1):
        question, reference = item["question"], item["answer"]
        logger.info(f"[baseline {i}/{len(items)}] {question[:70]}...")
        prediction = call_llm_no_context(rag, question)

        predictions.append(prediction)
        references.append(reference)
        judge_items.append({"question": question, "reference": reference, "answer": prediction})

    text_metrics = evaluate_batch(predictions, references)

    judge_summary = {}
    if run_judge:
        judge_result = judge_batch(judge_items, mode="baseline", judge_provider=rag.provider)
        judge_summary = {"relevancy": judge_result["relevancy"], "correctness": judge_result["correctness"]}

    summary = {**text_metrics, **judge_summary}

    output = {
        "summary": summary,
        "total_questions": len(predictions),
        "dataset_path": str(dataset_path),
        "provider": rag.provider,
        "results": judge_items,
    }

    output_path = output_path or Path(config["paths"]["logs_dir"]) / "baseline_evaluation_curated.json"
    save_json(output, str(output_path))

    logger.info("\n" + "=" * 45)
    logger.info("RESUMEN BASELINE (sin RAG)")
    logger.info("=" * 45)
    for k, v in summary.items():
        logger.info(f"  {k:15s}: {v:.4f}")
    logger.info("=" * 45)

    return output


def main():
    parser = argparse.ArgumentParser(description="Evaluación baseline sin RAG")
    parser.add_argument("--dataset", default="data/datasets/curated_eval_set.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-judge", action="store_true")
    args = parser.parse_args()

    config = load_config()
    run_baseline_evaluation(config, args.dataset, args.output, run_judge=not args.no_judge)


if __name__ == "__main__":
    main()
