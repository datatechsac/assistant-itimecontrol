"""
run_pipeline.py
Script maestro que ejecuta el pipeline completo de extremo a extremo.

Uso:
    # Pipeline completo
    python run_pipeline.py

    # Solo preprocesamiento
    python run_pipeline.py --stage preprocess

    # Solo RAG
    python run_pipeline.py --stage rag

    # Solo evaluación
    python run_pipeline.py --stage eval
"""
import argparse
import sys
from pathlib import Path

from src.utils.helpers import load_config, ensure_dirs
from src.utils.logger import get_logger

logger = get_logger(__name__, "logs/pipeline.log")


def run_preprocess(config: dict) -> bool:
    """Ejecuta las 4 etapas de preprocesamiento."""
    logger.info("\n" + "█" * 60)
    logger.info("PREPROCESAMIENTO")
    logger.info("█" * 60)
    try:
        from src.preprocessing.pdf_extractor import process_all_pdfs
        from src.preprocessing.text_cleaner  import process_all_texts
        from src.preprocessing.chunker       import process_all_texts as chunk_texts
        from src.preprocessing.dataset_builder import build_dataset_from_chunks

        results = process_all_pdfs(
            config["paths"]["raw_data"],
            config["paths"]["processed_data"],
        )
        if not results:
            logger.error("No se procesaron PDFs. ¿Colocaste los archivos en data/raw/?")
            return False

        process_all_texts(config["paths"]["processed_data"], config)
        chunk_texts(config["paths"]["processed_data"], config["paths"]["chunks_dir"], config)
        stats = build_dataset_from_chunks(
            config["paths"]["chunks_dir"],
            config["paths"]["datasets_dir"],
            config,
        )
        logger.info(f"Dataset: {stats.get('total', 0)} pares QA generados")
        return True

    except Exception as e:
        logger.error(f"Error en preprocesamiento: {e}", exc_info=True)
        return False


def run_format(config: dict) -> bool:
    """Formatea el dataset al formato de instrucción elegido."""
    logger.info("\n" + "█" * 60)
    logger.info("FORMATEO DE DATASET")
    logger.info("█" * 60)
    try:
        from src.fine_tuning.data_formatter import format_dataset
        from src.utils.helpers import load_jsonl, save_jsonl

        fmt = config["dataset"]["format"]
        datasets_dir = config["paths"]["datasets_dir"]

        for split in ["train", "val", "test"]:
            src = Path(datasets_dir) / f"{split}.jsonl"
            dst = Path(datasets_dir) / f"{split}_{fmt}.jsonl"
            if src.exists():
                records   = load_jsonl(str(src))
                formatted = format_dataset(records, fmt)
                save_jsonl(formatted, str(dst))
        return True
    except Exception as e:
        logger.error(f"Error formateando dataset: {e}", exc_info=True)
        return False


def run_finetune(config: dict) -> bool:
    """Ejecuta el fine-tuning del modelo."""
    logger.info("\n" + "█" * 60)
    logger.info("FINE-TUNING")
    logger.info("█" * 60)
    try:
        from src.fine_tuning.trainer import train
        train()
        return True
    except Exception as e:
        logger.error(f"Error en fine-tuning: {e}", exc_info=True)
        return False


def run_rag(config: dict) -> bool:
    """Genera embeddings e indexa en FAISS."""
    logger.info("\n" + "█" * 60)
    logger.info("INDEXACIÓN RAG")
    logger.info("█" * 60)
    try:
        from src.rag.embedder import load_chunks, generate_embeddings, build_faiss_index, save_index_and_metadata

        chunks = load_chunks(config["paths"]["chunks_dir"])
        if not chunks:
            return False

        texts      = [c["text"] for c in chunks]
        embeddings = generate_embeddings(texts, config["rag"]["embedding_model"])
        index      = build_faiss_index(embeddings)
        save_index_and_metadata(
            index, chunks,
            config["paths"]["embeddings_dir"],
            config["paths"]["vector_store"],
        )
        return True
    except Exception as e:
        logger.error(f"Error en indexación RAG: {e}", exc_info=True)
        return False


def run_eval(config: dict) -> bool:
    """Ejecuta el benchmark de evaluación."""
    logger.info("\n" + "█" * 60)
    logger.info("EVALUACIÓN")
    logger.info("█" * 60)
    try:
        from src.evaluation.benchmark import run_benchmark
        results = run_benchmark(config)
        return bool(results)
    except Exception as e:
        logger.error(f"Error en evaluación: {e}", exc_info=True)
        return False


STAGES = {
    "preprocess": run_preprocess,
    "format":     run_format,
    "finetune":   run_finetune,
    "rag":        run_rag,
    "eval":       run_eval,
}


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline completo iTimeControl Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages disponibles:
  preprocess  Extrae, limpia, chunka y construye el dataset
  format      Formatea el dataset al formato instrucción
  finetune    Entrena el modelo con LoRA/QLoRA
  rag         Genera embeddings e indexa en FAISS
  eval        Ejecuta el benchmark de evaluación
  all         Ejecuta todo el pipeline (por defecto)
        """,
    )
    parser.add_argument(
        "--stage", choices=list(STAGES.keys()) + ["all"],
        default="all",
        help="Etapa a ejecutar (default: all)",
    )
    args = parser.parse_args()

    config = load_config()
    ensure_dirs(config)

    logger.info("=" * 60)
    logger.info("iTimeControl Assistant — Pipeline maestro")
    logger.info("=" * 60)

    if args.stage == "all":
        pipeline = ["preprocess", "format", "finetune", "rag", "eval"]
    else:
        pipeline = [args.stage]

    results = {}
    for stage in pipeline:
        ok = STAGES[stage](config)
        results[stage] = "✓ OK" if ok else "✗ FALLÓ"
        if not ok and args.stage == "all":
            logger.error(f"Pipeline detenido en la etapa: {stage}")
            break

    logger.info("\n" + "=" * 60)
    logger.info("RESUMEN DEL PIPELINE")
    logger.info("=" * 60)
    for stage, status in results.items():
        logger.info(f"  {stage:12s}: {status}")
    logger.info("=" * 60)

    if any("FALLÓ" in v for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
