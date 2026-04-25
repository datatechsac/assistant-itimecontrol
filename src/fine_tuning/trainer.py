"""
fine_tuning/trainer.py
Entrenamiento del modelo con LoRA/QLoRA usando HuggingFace PEFT + TRL.

Uso:
    python src/fine_tuning/trainer.py

Requisitos de hardware:
  - GPU con al menos 8GB VRAM (ej. RTX 3080) para QLoRA 4-bit
  - CPU con 16GB RAM (entrenamiento muy lento)
"""
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from src.fine_tuning.config import load_training_config, TrainingConfig
from src.utils.helpers import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_tokenizer(cfg: TrainingConfig):
    """Carga y configura el tokenizador del modelo base."""
    logger.info(f"Cargando tokenizador: {cfg.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model,
        trust_remote_code=True,
        padding_side="right",
    )
    # Algunos modelos no tienen pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(cfg: TrainingConfig):
    """Carga el modelo base con cuantización 4-bit (QLoRA) si está disponible."""
    logger.info(f"Cargando modelo base: {cfg.base_model}")

    if cfg.load_in_4bit and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
        logger.info("Modelo cargado con cuantización QLoRA 4-bit")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )
        logger.warning("GPU no disponible. Entrenando en CPU (muy lento).")

    return model


def apply_lora(model, cfg: TrainingConfig):
    """Aplica la configuración LoRA al modelo base."""
    lora_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        target_modules=cfg.lora.target_modules,
        bias=cfg.lora.bias,
        task_type=cfg.lora.task_type,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_train_val_datasets(cfg: TrainingConfig):
    """Carga los datasets de entrenamiento y validación."""
    config = load_config()
    fmt = config["dataset"].get("format", "alpaca")
    datasets_dir = config["paths"]["datasets_dir"]

    train_path = str(Path(datasets_dir) / f"train_{fmt}.jsonl")
    val_path   = str(Path(datasets_dir) / f"val_{fmt}.jsonl")

    # Si no existen los formateados, usar los originales
    if not Path(train_path).exists():
        train_path = cfg.train_file
        val_path = cfg.val_file
        logger.warning("Usando datasets sin formatear. Ejecuta data_formatter.py primero.")

    train_dataset = load_dataset("json", data_files=train_path, split="train")
    val_dataset   = load_dataset("json", data_files=val_path,   split="train")

    logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    return train_dataset, val_dataset


def get_training_arguments(cfg: TrainingConfig) -> TrainingArguments:
    """Crea los TrainingArguments de HuggingFace."""
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    return TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        weight_decay=cfg.weight_decay,
        fp16=cfg.fp16,
        bf16=cfg.bf16 and torch.cuda.is_bf16_supported(),
        logging_steps=cfg.logging_steps,
        evaluation_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",   # cambiar a "wandb" si usas Weights & Biases
        seed=cfg.seed,
        max_grad_norm=cfg.max_grad_norm,
        dataloader_num_workers=0,
    )


def train(cfg: TrainingConfig | None = None):
    """Pipeline completo de fine-tuning."""
    if cfg is None:
        cfg = load_training_config()

    logger.info("=" * 60)
    logger.info("FINE-TUNING: iTimeControl Assistant")
    logger.info("=" * 60)

    tokenizer = load_tokenizer(cfg)
    model = load_base_model(cfg)
    model = apply_lora(model, cfg)

    train_dataset, val_dataset = load_train_val_datasets(cfg)
    training_args = get_training_arguments(cfg)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=cfg.tokenizer_max_length,
        packing=False,
    )

    logger.info("Iniciando entrenamiento...")
    trainer.train()

    # Guardar modelo final
    final_path = Path(cfg.output_dir) / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    logger.info(f"Modelo guardado en: {final_path}")

    return trainer


def main():
    cfg = load_training_config()
    train(cfg)
    logger.info("\nEntrenamiento completado.")
    logger.info("Siguiente paso: python src/rag/embedder.py")


if __name__ == "__main__":
    main()
