"""
rag/pipeline.py
Pipeline RAG: FAISS retriever + LLM para generación de respuestas.

Proveedores soportados (config.yaml → generation.provider):
  - "groq"      → Llama 3.3 70B via Groq API (GRATUITO)
  - "gemini"    → Gemini 2.0 Flash via Google AI Studio (GRATUITO)
  - "anthropic" → Claude Haiku (pago)

Variables de entorno requeridas según proveedor:
  - GROQ_API_KEY      → console.groq.com/keys
  - GEMINI_API_KEY    → aistudio.google.com/apikey
  - ANTHROPIC_API_KEY → console.anthropic.com
"""
import os

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
    """Pipeline RAG multi-proveedor: FAISS retriever + LLM configurable."""

    def __init__(self, config: dict | None = None):
        if config is None:
            config = load_config()

        self.config   = config
        self.provider = config.get("generation", {}).get("provider", "groq")
        self.retriever = Retriever(config)
        self.client   = self._build_client()
        logger.info(f"RAG Pipeline listo (proveedor: {self.provider})")

    def _build_client(self):
        """Inicializa el cliente del proveedor configurado."""
        if self.provider == "groq":
            return self._init_groq()
        if self.provider == "gemini":
            return self._init_gemini()
        if self.provider == "anthropic":
            return self._init_anthropic()
        raise ValueError(
            f"Proveedor no soportado: '{self.provider}'. "
            "Usa 'groq', 'gemini' o 'anthropic' en config.yaml → generation.provider"
        )

    def _init_groq(self):
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("Instala groq: pip install groq")

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY no encontrada.\n"
                "Obtén tu clave gratis en: https://console.groq.com/keys\n"
                "Luego: set GROQ_API_KEY=gsk_..."
            )
        cfg = self.config.get("groq", {})
        self._model     = cfg.get("model", "llama-3.3-70b-versatile")
        self._max_tokens = cfg.get("max_tokens", 512)
        self._temperature = cfg.get("temperature", 0.3)
        logger.info(f"Groq cliente listo (modelo: {self._model})")
        return Groq(api_key=api_key)

    def _init_gemini(self):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Instala google-generativeai: pip install google-generativeai")

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY no encontrada.\n"
                "Obtén tu clave gratis en: https://aistudio.google.com/apikey\n"
                "Luego: set GEMINI_API_KEY=AIza..."
            )
        cfg = self.config.get("gemini", {})
        self._model      = cfg.get("model", "gemini-2.0-flash")
        self._max_tokens = cfg.get("max_tokens", 512)
        self._temperature = cfg.get("temperature", 0.3)
        genai.configure(api_key=api_key)
        logger.info(f"Gemini cliente listo (modelo: {self._model})")
        return genai.GenerativeModel(
            model_name=self._model,
            system_instruction=SYSTEM_PROMPT,
            generation_config=genai.GenerationConfig(
                max_output_tokens=self._max_tokens,
                temperature=self._temperature,
            ),
        )

    def _init_anthropic(self):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Instala anthropic: pip install anthropic")

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY no encontrada.\n"
                "Configúrala en: https://console.anthropic.com\n"
                "Luego: set ANTHROPIC_API_KEY=sk-ant-..."
            )
        cfg = self.config.get("claude", {})
        self._model      = cfg.get("model", "claude-haiku-4-5-20251001")
        self._max_tokens = cfg.get("max_tokens", 512)
        self._temperature = cfg.get("temperature", 0.3)
        logger.info(f"Anthropic cliente listo (modelo: {self._model})")
        return anthropic.Anthropic(api_key=api_key)

    def _call_llm(self, user_message: str) -> str:
        """Llama al LLM configurado y devuelve el texto de la respuesta."""
        if self.provider == "groq":
            resp = self.client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
            )
            return resp.choices[0].message.content.strip()

        if self.provider == "gemini":
            resp = self.client.generate_content(user_message)
            return resp.text.strip()

        if self.provider == "anthropic":
            resp = self.client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            return resp.content[0].text.strip()

        raise ValueError(f"Proveedor no soportado: {self.provider}")

    def generate(self, question: str) -> dict:
        """
        Recupera contexto relevante y genera una respuesta con el LLM.

        Returns:
            Dict con 'answer', 'context_used', 'contexts', 'sources', 'num_chunks'.
        """
        retrieved = self.retriever.search(question)
        context   = self.retriever.format_context(retrieved)

        user_message = (
            f"Contexto de iTimeControl:\n{context}\n\nPregunta: {question}"
            if context else question
        )

        answer  = self._call_llm(user_message)
        sources = list({r["source"] for r in retrieved})

        return {
            "answer":       answer,
            "context_used": context,
            "contexts":     [r["text"] for r in retrieved],
            "sources":      sources,
            "num_chunks":   len(retrieved),
        }


def interactive_demo() -> None:
    """Modo demo interactivo en terminal."""
    config = load_config()
    rag    = RAGPipeline(config)
    provider = config.get("generation", {}).get("provider", "groq")

    logger.info("\n" + "=" * 60)
    logger.info(f"iTimeControl Assistant — Demo ({provider})")
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
        print(f"\nFuentes : {', '.join(result['sources']) or 'N/A'}")
        print(f"Chunks  : {result['num_chunks']}\n")
        print("-" * 60)


def main() -> None:
    interactive_demo()


if __name__ == "__main__":
    main()
