"""
fine_tuning/model_saver.py
Utilidades para guardar, cargar y fusionar adaptadores LoRA con el modelo base.

Uso:
    python src/fine_tuning/model_saver.py --merge
"""
import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.fine_tuning.config import load_training_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_adapter(model, tokenizer, output_dir: str) -> None:
    """
    Guarda solo el adaptador LoRA (ligero, recomendado durante desarrollo).

    Args:
        model: Modelo PEFT con adaptador LoRA.
        tokenizer: Tokenizador del modelo.
        output_dir: Directorio de salida.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Adaptador LoRA guardado en: {output_dir}")


def merge_and_save(adapter_dir: str, output_dir: str, base_model: str | None = None) -> None:
    """
    Fusiona el adaptador LoRA con el modelo base y guarda el modelo completo.
    Útil para deployment sin dependencia de PEFT.

    Args:
        adapter_dir: Directorio con el adaptador LoRA guardado.
        output_dir:  Directorio donde guardar el modelo fusionado.
        base_model:  Nombre del modelo base (si None, lee desde config).
    """
    if base_model is None:
        cfg = load_training_config()
        base_model = cfg.base_model

    logger.info(f"Cargando modelo base: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info(f"Cargando adaptador LoRA: {adapter_dir}")
    model = PeftModel.from_pretrained(base, adapter_dir)

    logger.info("Fusionando adaptador con modelo base...")
    model = model.merge_and_unload()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Modelo fusionado guardado en: {output_dir}")


def list_checkpoints(checkpoints_dir: str) -> list[str]:
    """Lista todos los checkpoints disponibles ordenados por step."""
    ckpts = sorted(
        Path(checkpoints_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
    )
    return [str(c) for c in ckpts]


def main():
    parser = argparse.ArgumentParser(description="Gestión del modelo fine-tuned")
    parser.add_argument(
        "--merge", action="store_true",
        help="Fusiona el adaptador LoRA con el modelo base"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Lista los checkpoints disponibles"
    )
    args = parser.parse_args()

    cfg = load_training_config()

    if args.list:
        checkpoints = list_checkpoints(cfg.output_dir)
        if checkpoints:
            logger.info(f"Checkpoints disponibles ({len(checkpoints)}):")
            for ck in checkpoints:
                logger.info(f"  {ck}")
        else:
            logger.info("No hay checkpoints disponibles.")

    elif args.merge:
        adapter_dir = str(Path(cfg.output_dir) / "final")
        merged_dir  = str(Path(cfg.output_dir) / "merged")
        merge_and_save(adapter_dir, merged_dir, cfg.base_model)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
