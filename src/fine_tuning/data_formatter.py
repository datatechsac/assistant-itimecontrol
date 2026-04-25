"""
fine_tuning/data_formatter.py
Convierte el dataset JSONL al formato de instrucción requerido por el modelo.

Formatos soportados:
  - alpaca: ### Instruction / ### Input / ### Response
  - chatml: <|im_start|>system ... <|im_end|>
  - llama2: [INST] ... [/INST]

Uso:
    python src/fine_tuning/data_formatter.py
"""
from pathlib import Path

from datasets import Dataset

from src.fine_tuning.config import load_training_config
from src.utils.helpers import load_jsonl, save_jsonl, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "Eres un asistente experto en el sistema iTimeControl. "
    "Responde de forma clara, precisa y en español. "
    "Si no conoces la respuesta, indícalo honestamente."
)


def format_alpaca(record: dict) -> str:
    """Formato Alpaca con separadores de sección."""
    instruction = record.get("instruction", "")
    input_text = record.get("input", "").strip()
    output = record.get("output", "")

    if input_text:
        return (
            f"### Instrucción:\n{instruction}\n\n"
            f"### Contexto:\n{input_text}\n\n"
            f"### Respuesta:\n{output}"
        )
    return (
        f"### Instrucción:\n{instruction}\n\n"
        f"### Respuesta:\n{output}"
    )


def format_chatml(record: dict) -> str:
    """Formato ChatML compatible con Mistral/Qwen/Phi."""
    instruction = record.get("instruction", "")
    input_text = record.get("input", "").strip()
    output = record.get("output", "")

    user_message = f"{instruction}\n{input_text}".strip() if input_text else instruction

    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_message}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )


def format_llama2(record: dict) -> str:
    """Formato LLaMA 2 Chat."""
    instruction = record.get("instruction", "")
    input_text = record.get("input", "").strip()
    output = record.get("output", "")

    user_message = f"{instruction}\n{input_text}".strip() if input_text else instruction

    return (
        f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"{user_message} [/INST] {output} </s>"
    )


FORMATTERS = {
    "alpaca": format_alpaca,
    "chatml": format_chatml,
    "llama2": format_llama2,
}


def format_dataset(records: list[dict], fmt: str) -> list[dict]:
    """
    Aplica el formatter seleccionado a todos los registros.

    Returns:
        Lista de dicts con campo 'text' listo para el entrenamiento.
    """
    formatter = FORMATTERS.get(fmt, format_alpaca)
    formatted = []

    for record in records:
        try:
            text = formatter(record)
            if len(text.strip()) > 20:
                formatted.append({
                    "text": text,
                    "source": record.get("source", ""),
                    "chunk_id": record.get("chunk_id", ""),
                })
        except Exception as e:
            logger.warning(f"Error formateando registro: {e}")

    return formatted


def load_as_hf_dataset(jsonl_path: str, fmt: str) -> Dataset:
    """
    Carga un JSONL y lo retorna como HuggingFace Dataset formateado.

    Args:
        jsonl_path: Ruta al archivo JSONL.
        fmt: Formato de instrucción ('alpaca', 'chatml', 'llama2').

    Returns:
        HuggingFace Dataset con columna 'text'.
    """
    records = load_jsonl(jsonl_path)
    formatted = format_dataset(records, fmt)
    return Dataset.from_list(formatted)


def main():
    cfg = load_training_config()
    config = load_config()
    datasets_dir = config["paths"]["datasets_dir"]
    fmt = cfg.dataset_format

    logger.info("=" * 60)
    logger.info(f"Formateando datasets con formato: {fmt.upper()}")
    logger.info("=" * 60)

    for split in ["train", "val", "test"]:
        src = Path(datasets_dir) / f"{split}.jsonl"
        dst = Path(datasets_dir) / f"{split}_{fmt}.jsonl"

        if not src.exists():
            logger.warning(f"No encontrado: {src}")
            continue

        records = load_jsonl(str(src))
        formatted = format_dataset(records, fmt)
        save_jsonl(formatted, str(dst))

        logger.info(f"  ✓ {split}: {len(records)} → {len(formatted)} registros → {dst.name}")

    logger.info("\nFormato completado. Siguiente paso: python src/fine_tuning/trainer.py")


if __name__ == "__main__":
    main()
