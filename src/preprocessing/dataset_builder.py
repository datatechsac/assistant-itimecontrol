"""
preprocessing/dataset_builder.py
Genera pares Pregunta-Respuesta (QA) desde los chunks para el fine-tuning.

Estrategia:
  - Genera preguntas basadas en keywords y patrones del dominio iTimeControl.
  - Crea el dataset en formato Alpaca (instruction, input, output).
  - Divide en train/val/test según config.yaml.

Uso:
    python src/preprocessing/dataset_builder.py
"""
import random
from pathlib import Path

from src.utils.helpers import load_config, ensure_dirs, load_json, save_jsonl
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Plantillas de preguntas para el dominio iTimeControl
QUESTION_TEMPLATES = [
    "¿Cómo {action} en iTimeControl?",
    "¿Cuál es el procedimiento para {action}?",
    "Explica cómo {action} en el sistema.",
    "¿Qué pasos debo seguir para {action}?",
    "¿Dónde puedo {action} en iTimeControl?",
    "¿Cuál es la función de {entity} en iTimeControl?",
    "Describe el proceso de {action}.",
    "¿Cómo funciona {entity} en el sistema iTimeControl?",
]

# Keywords del dominio
DOMAIN_KEYWORDS = [
    "registrar asistencia", "marcar entrada", "marcar salida",
    "gestionar empleados", "configurar horarios", "ver reportes",
    "exportar datos", "asignar turnos", "calcular horas extras",
    "gestionar permisos", "solicitar vacaciones", "aprobar solicitudes",
    "generar nómina", "configurar alertas", "administrar usuarios",
]


def chunk_to_qa_pair(chunk: dict) -> dict | None:
    """
    Convierte un chunk de texto en un par QA para fine-tuning.

    Usa el texto del chunk como respuesta y genera una pregunta
    contextual basada en el contenido.

    Args:
        chunk: Dict con 'text' y 'source'.

    Returns:
        Dict con instruction/input/output o None si el chunk es muy corto.
    """
    text = chunk.get("text", "").strip()

    if len(text.split()) < 30:
        return None

    # Detectar keywords del dominio en el chunk
    text_lower = text.lower()
    matched_keywords = [kw for kw in DOMAIN_KEYWORDS if kw in text_lower]

    if matched_keywords:
        keyword = random.choice(matched_keywords)
        # Determinar si es acción o entidad
        if any(verb in keyword for verb in ["registrar", "marcar", "gestionar",
                                             "configurar", "ver", "exportar",
                                             "asignar", "calcular", "generar"]):
            template = random.choice([t for t in QUESTION_TEMPLATES if "{action}" in t])
            question = template.format(action=keyword)
        else:
            template = random.choice([t for t in QUESTION_TEMPLATES if "{entity}" in t])
            question = template.format(entity=keyword)
    else:
        # Pregunta genérica sobre el fragmento
        first_sentence = text.split(".")[0][:100]
        question = f"¿Puedes explicar lo siguiente sobre iTimeControl: {first_sentence}?"

    return {
        "instruction": question,
        "input": "",
        "output": text,
        "source": chunk.get("source", ""),
        "chunk_id": chunk.get("id", ""),
    }


def build_dataset_from_chunks(chunks_dir: str, datasets_dir: str, config: dict) -> dict:
    """
    Construye el dataset completo de fine-tuning desde los chunks.

    Returns:
        Dict con estadísticas del dataset generado.
    """
    master_file = Path(chunks_dir) / "all_chunks.json"

    if not master_file.exists():
        logger.error(f"No se encontró: {master_file}")
        logger.error("Ejecuta primero: python src/preprocessing/chunker.py")
        return {}

    chunks = load_json(str(master_file))
    logger.info(f"Chunks cargados: {len(chunks)}")

    # Generar pares QA
    qa_pairs = []
    skipped = 0
    for chunk in chunks:
        pair = chunk_to_qa_pair(chunk)
        if pair:
            qa_pairs.append(pair)
        else:
            skipped += 1

    if not qa_pairs:
        logger.error("No se generaron pares QA. Verifica que los chunks tengan contenido suficiente.")
        return {}

    logger.info(f"Pares QA generados: {len(qa_pairs)} (omitidos: {skipped})")

    # Mezclar para evitar orden de fuente
    random.seed(config["training"].get("seed", 42))
    random.shuffle(qa_pairs)

    # Dividir en splits
    total = len(qa_pairs)
    train_ratio = config["dataset"].get("train_ratio", 0.85)
    val_ratio = config["dataset"].get("val_ratio", 0.10)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train": qa_pairs[:train_end],
        "val": qa_pairs[train_end:val_end],
        "test": qa_pairs[val_end:],
    }

    Path(datasets_dir).mkdir(parents=True, exist_ok=True)

    for split_name, records in splits.items():
        out_path = Path(datasets_dir) / f"{split_name}.jsonl"
        save_jsonl(records, str(out_path))
        logger.info(f"  {split_name}: {len(records)} registros → {out_path}")

    # Guardar preguntas de benchmark (del test set)
    benchmark = [{"question": r["instruction"], "answer": r["output"]}
                 for r in splits["test"][:50]]
    from src.utils.helpers import save_json
    save_json(benchmark, str(Path(datasets_dir) / "benchmark_questions.json"))

    return {
        "total": total,
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
    }


def main():
    config = load_config()
    ensure_dirs(config)

    chunks_dir = config["paths"]["chunks_dir"]
    datasets_dir = config["paths"]["datasets_dir"]

    logger.info("=" * 60)
    logger.info("ETAPA 4: Construcción del dataset de fine-tuning")
    logger.info("=" * 60)

    stats = build_dataset_from_chunks(chunks_dir, datasets_dir, config)

    if stats:
        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset listo: {stats['total']} pares QA totales")
        logger.info(f"  Train: {stats['train']} | Val: {stats['val']} | Test: {stats['test']}")
        logger.info("Siguiente paso: python src/fine_tuning/data_formatter.py")


if __name__ == "__main__":
    main()
