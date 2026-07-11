"""
rag/pipeline.py
Pipeline RAG: recupera contexto con FAISS + genera respuesta con Claude API.
No requiere GPU ni modelo local.
"""
import os
from pathlib import Path

import anthropic

from src.rag.retriever import Retriever
from src.utils.helpers import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "Eres un asistente experto en el sistema iTimeControl. "
    "Usa únicamente la información del contexto provisto para responder con precisión y en español. "
    "Si la información no es suficiente para responder, indícalo claramente."
)


class RAGPipeline:
    """Pipeline RAG: FAISS retriever + Claude API para generación."""

    def __init__(self, config: dict | None = None):
        if config is None:
            config = load_config()

        self.config = config
        claude_cfg = config.get("claude", {})
        self.model = claude_cfg.get("model", "claude-haiku-4-5-20251001")
        self.max_tokens = claude_cfg.get("max_tokens", 512)
        self.temperature = claude_cfg.get("temperature", 0.3)

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY no encontrada. "
                "Ejecuta: export ANTHROPIC_API_KEY=sk-ant-..."
            )

        self.client = anthropic.Anthropic(api_key=api_key)
        self.retriever = Retriever(config)
        logger.info(f"RAG Pipeline listo (modelo: {self.model})")

    def generate(self, question: str) -> dict:
        """
        Recupera contexto relevante y genera una respuesta con Claude.

        Returns:
            Dict con 'answer', 'context_used', 'contexts', 'sources', 'num_chunks'.
        """
        retrieved = self.retriever.search(question)
        context = self.retriever.format_context(retrieved)

        user_message = (
            f"Contexto de iTimeControl:\n{context}\n\nPregunta: {question}"
            if context else question
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        answer = response.content[0].text.strip()
        sources = list({r["source"] for r in retrieved})

        return {
            "answer": answer,
            "context_used": context,
            "contexts": [r["text"] for r in retrieved],
            "sources": sources,
            "num_chunks": len(retrieved),
        }


def interactive_demo() -> None:
    """Modo demo interactivo en terminal."""
    config = load_config()
    rag = RAGPipeline(config)

    logger.info("\n" + "=" * 60)
    logger.info("iTimeControl Assistant — Modo demo (Claude API)")
    logger.info("Escribe 'salir' para terminar")
    logger.info("=" * 60 + "\n")

    while True:
        question = input("Tu pregunta: ").strip()
        if question.lower() in {"salir", "exit", "quit"}:
            break
        if not question:
            continue

        result = rag.generate(question)
        print(f"\nRespuesta:\n{result['answer']}")
        print(f"\nFuentes: {', '.join(result['sources']) or 'N/A'}")
        print(f"Chunks usados: {result['num_chunks']}\n")
        print("-" * 60)


def main() -> None:
    interactive_demo()


if __name__ == "__main__":
    main()
