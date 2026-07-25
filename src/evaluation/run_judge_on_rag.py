"""
evaluation/run_judge_on_rag.py
Aplica LLM-as-judge (faithfulness, relevancy, correctness) sobre resultados
ya generados por src/evaluation/rag_eval.py (evita volver a llamar al retriever/LLM).

Uso:
    python -m src.evaluation.run_judge_on_rag --input logs/rag_evaluation_curated.json
"""
import argparse
from pathlib import Path

from src.evaluation.llm_judge import judge_batch
from src.utils.helpers import load_config, load_json, save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_judge_on_rag_results(
    config: dict,
    input_path: str,
    output_path: str | None = None,
    judge_provider: str | None = None,
) -> dict:
    data = load_json(input_path)
    raw_results = data["results"]

    items = []
    for r in raw_results:
        # Usa solo los chunks efectivamente pasados al LLM en la generación (num_chunks)
        used_contexts = r["contexts"][: r.get("num_chunks", len(r["contexts"]))]
        context_text = "\n\n".join(used_contexts)
        items.append({
            "question": r["question"],
            "context": context_text,
            "reference": r["reference"],
            "answer": r["prediction"],
        })

    provider = judge_provider or data.get("provider", config.get("generation", {}).get("provider", "groq"))
    judge_result = judge_batch(items, mode="rag", judge_provider=provider)

    summary = {
        "faithfulness": judge_result["faithfulness"],
        "relevancy": judge_result["relevancy"],
        "correctness": judge_result["correctness"],
    }

    output = {
        "summary": summary,
        "total_questions": len(items),
        "source_file": input_path,
        "judge_provider": judge_result["judge_provider"],
        "judge_model": judge_result["judge_model"],
        "results": judge_result["details"],
    }

    output_path = output_path or Path(config["paths"]["logs_dir"]) / "rag_judge_curated.json"
    save_json(output, str(output_path))

    logger.info("\n" + "=" * 45)
    logger.info("RESUMEN LLM-JUDGE (RAG)")
    logger.info("=" * 45)
    for k, v in summary.items():
        logger.info(f"  {k:15s}: {v:.4f}")
    logger.info("=" * 45)

    return output


def main():
    parser = argparse.ArgumentParser(description="LLM-judge sobre resultados RAG ya generados")
    parser.add_argument("--input", default="logs/rag_evaluation_curated.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--provider", default=None, help="Proveedor del juez: groq | gemini | anthropic")
    args = parser.parse_args()

    config = load_config()
    run_judge_on_rag_results(config, args.input, args.output, judge_provider=args.provider)


if __name__ == "__main__":
    main()
