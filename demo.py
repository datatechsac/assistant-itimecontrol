"""
demo.py — iTimeControl Assistant Demo
Ejecuta preguntas predefinidas o modo interactivo contra el pipeline RAG completo.

Uso:
    python demo.py                  # preguntas predefinidas
    python demo.py --interactive    # modo interactivo en terminal
    python demo.py --solo-retriever # solo muestra chunks recuperados (sin LLM)
"""
import argparse
import time
from pathlib import Path

# ── Preguntas de demo para iTimeControl ──────────────────────────────────────
DEMO_QUESTIONS = [
    "¿Cómo registro la asistencia de un empleado en iTimeControl?",
    "¿Cuántas horas extras puede trabajar un empleado según la ley peruana?",
    "¿Cómo genero un reporte de horas trabajadas del mes?",
    "¿Qué documentos necesito para gestionar permisos laborales?",
    "¿Cómo configuro los turnos de trabajo en el sistema?",
]

SEP = "─" * 65


def print_header():
    print()
    print("=" * 65)
    print("   iTimeControl Assistant — Demo RAG + Fine-Tuning")
    print("=" * 65)
    print()


def run_retriever_only(questions: list[str]):
    """Demo ligero: solo muestra los chunks recuperados (sin LLM)."""
    from src.rag.retriever import Retriever
    from src.utils.helpers import load_config

    config = load_config()
    retriever = Retriever(config)

    print_header()
    print(f"Modo: Solo Retriever FAISS ({retriever.index.ntotal} vectores indexados)\n")

    for i, question in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {question}")
        print(SEP)

        t0 = time.time()
        results = retriever.search(question, top_k=3)
        elapsed = (time.time() - t0) * 1000

        for j, r in enumerate(results, 1):
            source = Path(r["source"]).stem if r["source"] else "N/A"
            preview = r["text"][:200].replace("\n", " ")
            print(f"  Chunk {j} | score={r['score']:.3f} | {source}")
            print(f"  {preview}...")
            print()

        print(f"  Latencia retriever: {elapsed:.1f} ms")
        print()


def run_full_pipeline(questions: list[str]):
    """Demo completo: retriever FAISS + generación con Mistral-7B fine-tuned."""
    from src.rag.pipeline import RAGPipeline
    from src.utils.helpers import load_config

    print_header()
    print("Cargando pipeline RAG (retriever + Mistral-7B + LoRA)...")
    print("Esto puede tomar 1-2 minutos la primera vez.\n")

    config = load_config()
    rag = RAGPipeline(config)

    print(f"\nPipeline listo. Ejecutando {len(questions)} preguntas de demo.\n")
    print(SEP)

    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] PREGUNTA:")
        print(f"  {question}")
        print()

        t0 = time.time()
        result = rag.generate(question)
        elapsed = time.time() - t0

        print("RESPUESTA:")
        for line in result["answer"].splitlines():
            print(f"  {line}")
        print()
        sources_str = ", ".join(Path(s).stem for s in result["sources"]) or "N/A"
        print(f"  Fuentes      : {sources_str}")
        print(f"  Chunks usados: {result['num_chunks']}")
        print(f"  Tiempo total : {elapsed:.1f}s")
        print(SEP)

    print("\nDemo finalizada.")


def run_interactive():
    """Modo interactivo: el usuario escribe sus propias preguntas."""
    from src.rag.pipeline import RAGPipeline
    from src.utils.helpers import load_config

    print_header()
    print("Cargando pipeline RAG...")

    config = load_config()
    rag = RAGPipeline(config)

    print("\nPipeline listo.")
    print("Escribe tu pregunta sobre iTimeControl (o 'salir' para terminar).\n")

    while True:
        try:
            question = input("Tu pregunta: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question:
            continue
        if question.lower() in {"salir", "exit", "quit", "q"}:
            break

        print()
        t0 = time.time()
        result = rag.generate(question)
        elapsed = time.time() - t0

        print("Respuesta:")
        for line in result["answer"].splitlines():
            print(f"  {line}")
        sources_str = ", ".join(Path(s).stem for s in result["sources"]) or "N/A"
        print(f"\n  Fuentes: {sources_str} | Chunks: {result['num_chunks']} | {elapsed:.1f}s")
        print(SEP + "\n")

    print("Sesión terminada.")


def main():
    parser = argparse.ArgumentParser(
        description="Demo del Asistente iTimeControl (RAG + Fine-Tuning)"
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Modo interactivo: escribe tus propias preguntas"
    )
    parser.add_argument(
        "--solo-retriever", action="store_true",
        help="Solo muestra chunks recuperados, sin generación LLM"
    )
    parser.add_argument(
        "--preguntas", nargs="+",
        help="Preguntas personalizadas para la demo (reemplaza las predefinidas)"
    )
    args = parser.parse_args()

    questions = args.preguntas if args.preguntas else DEMO_QUESTIONS

    if args.interactive:
        run_interactive()
    elif args.solo_retriever:
        run_retriever_only(questions)
    else:
        run_full_pipeline(questions)


if __name__ == "__main__":
    main()
