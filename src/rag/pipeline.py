"""
rag/pipeline.py
Pipeline RAG completo: recupera contexto + genera respuesta con el modelo fine-tuned.

Uso:
    python src/rag/pipeline.py
    (modo interactivo de prueba)
"""
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, pipeline

from src.rag.retriever import Retriever
from src.utils.helpers import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "Eres un asistente experto en el sistema iTimeControl. "
    "Usa la información de contexto provista para responder con precisión. "
    "Si la información no es suficiente, indícalo."
)

_USE_4BIT = not torch.cuda.is_available()


class RAGPipeline:
    """
    Pipeline que combina recuperación semántica (RAG) con el modelo fine-tuned.
    """

    def __init__(self, config: dict | None = None):
        if config is None:
            config = load_config()

        self.config = config
        self.api_cfg = config.get("api", {})

        # Cargar retriever
        self.retriever = Retriever(config)

        # Cargar modelo fine-tuned
        model_source, is_adapter = self._resolve_model_dir(config)
        logger.info(f"Cargando modelo desde: {model_source} (adapter={is_adapter})")

        quant_cfg = (
            BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            if _USE_4BIT else None
        )
        load_kwargs = dict(
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True,
        )
        if quant_cfg:
            load_kwargs["quantization_config"] = quant_cfg

        if is_adapter:
            from peft import PeftModel
            base_model_id = config["model"]["base_model"]
            self.tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
            base = AutoModelForCausalLM.from_pretrained(base_model_id, **load_kwargs)
            self.model = PeftModel.from_pretrained(base, model_source)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_source, **load_kwargs)

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        logger.info("RAG Pipeline listo.")

    def _resolve_model_dir(self, config: dict) -> tuple[str, bool]:
        """
        Devuelve (ruta, is_adapter).
        is_adapter=True  → directorio con adapter LoRA (requiere PEFT).
        is_adapter=False → modelo completo/merged o modelo base.
        """
        checkpoints_dir = Path(config["paths"]["models_dir"])

        # Prioridad 1: modelo fusionado listo para inferencia directa
        merged_dir = checkpoints_dir / "merged"
        if merged_dir.exists():
            return str(merged_dir), False

        # Prioridad 2: adapter LoRA — se carga con PeftModel sobre el modelo base
        final_dir = checkpoints_dir / "final"
        if final_dir.exists():
            logger.info("Cargando adapter LoRA desde 'final/' con PEFT.")
            return str(final_dir), True

        # Prioridad 3: último checkpoint disponible
        checkpoints = sorted(checkpoints_dir.glob("checkpoint-*"))
        if checkpoints:
            logger.warning(f"Usando checkpoint: {checkpoints[-1]}")
            return str(checkpoints[-1]), True

        # Fallback al modelo base sin fine-tuning
        base_model = config["model"]["base_model"]
        logger.warning(f"No se encontró modelo fine-tuned. Usando base: {base_model}")
        return base_model, False

    def build_prompt(self, question: str, context: str) -> str:
        """Construye el prompt con contexto RAG en formato ChatML."""
        user_message = (
            f"Contexto de iTimeControl:\n{context}\n\n"
            f"Pregunta: {question}"
        ) if context else question

        return (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_message}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def generate(self, question: str) -> dict:
        """
        Genera una respuesta a partir de una pregunta.

        Args:
            question: Pregunta del usuario sobre iTimeControl.

        Returns:
            Dict con 'answer', 'context_used', 'sources', 'num_chunks'.
        """
        # Recuperar contexto relevante
        retrieved = self.retriever.search(question)
        context = self.retriever.format_context(retrieved)

        # Construir prompt
        prompt = self.build_prompt(question, context)

        # Generar respuesta
        gen_config = GenerationConfig(
            max_new_tokens=self.api_cfg.get("max_new_tokens", 512),
            temperature=self.api_cfg.get("temperature", 0.7),
            top_p=self.api_cfg.get("top_p", 0.9),
            repetition_penalty=self.api_cfg.get("repetition_penalty", 1.1),
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        output = self.generator(prompt, generation_config=gen_config)

        # Extraer solo la respuesta generada (sin el prompt)
        generated_text = output[0]["generated_text"]
        answer = generated_text[len(prompt):].strip()

        # Limpiar token de fin si aparece
        for end_token in ["<|im_end|>", "</s>", "[/INST]"]:
            if end_token in answer:
                answer = answer.split(end_token)[0].strip()

        sources = list({r["source"] for r in retrieved})

        return {
            "answer": answer,
            "context_used": context,
            "sources": sources,
            "num_chunks": len(retrieved),
        }


def interactive_demo():
    """Modo demo interactivo en terminal."""
    config = load_config()
    rag = RAGPipeline(config)

    logger.info("\n" + "=" * 60)
    logger.info("iTimeControl Assistant — Modo demo")
    logger.info("Escribe 'salir' para terminar")
    logger.info("=" * 60 + "\n")

    while True:
        question = input("🙋 Tu pregunta: ").strip()
        if question.lower() in {"salir", "exit", "quit"}:
            break
        if not question:
            continue

        result = rag.generate(question)
        print(f"\n🤖 Respuesta:\n{result['answer']}")
        print(f"\n📄 Fuentes: {', '.join(result['sources']) or 'N/A'}")
        print(f"📊 Chunks usados: {result['num_chunks']}\n")
        print("-" * 60)


def main():
    interactive_demo()


if __name__ == "__main__":
    main()
