"""
fine_tuning/config.py
Carga y expone la configuración de entrenamiento desde config.yaml.
"""
from dataclasses import dataclass, field
from pathlib import Path

from src.utils.helpers import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    output_dir: str = "models/checkpoints"
    load_in_4bit: bool = True
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.001
    fp16: bool = False
    bf16: bool = True
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100
    max_grad_norm: float = 0.3
    seed: int = 42
    tokenizer_max_length: int = 2048
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    train_file: str = "data/datasets/train.jsonl"
    val_file: str = "data/datasets/val.jsonl"
    dataset_format: str = "alpaca"


def load_training_config() -> TrainingConfig:
    """Carga TrainingConfig desde config.yaml."""
    cfg = load_config()
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    lora_cfg = cfg.get("lora", {})
    dataset_cfg = cfg.get("dataset", {})
    paths_cfg = cfg.get("paths", {})

    lora = LoRAConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )

    training = TrainingConfig(
        base_model=model_cfg.get("base_model", "mistralai/Mistral-7B-Instruct-v0.2"),
        output_dir=model_cfg.get("output_dir", "models/checkpoints"),
        load_in_4bit=model_cfg.get("load_in_4bit", True),
        tokenizer_max_length=model_cfg.get("tokenizer_max_length", 2048),
        num_epochs=train_cfg.get("num_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 2),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        weight_decay=train_cfg.get("weight_decay", 0.001),
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", True),
        logging_steps=train_cfg.get("logging_steps", 10),
        eval_steps=train_cfg.get("eval_steps", 50),
        save_steps=train_cfg.get("save_steps", 100),
        max_grad_norm=train_cfg.get("max_grad_norm", 0.3),
        seed=train_cfg.get("seed", 42),
        train_file=str(Path(paths_cfg.get("datasets_dir", "data/datasets")) / "train.jsonl"),
        val_file=str(Path(paths_cfg.get("datasets_dir", "data/datasets")) / "val.jsonl"),
        dataset_format=dataset_cfg.get("format", "alpaca"),
        lora=lora,
    )

    logger.info(f"Configuración de entrenamiento cargada: modelo={training.base_model}")
    return training
