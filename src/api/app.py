"""
api/app.py
Interfaz web con Gradio para el asistente iTimeControl.

Uso:
    python src/api/app.py
"""
import gradio as gr

from src.rag.pipeline import RAGPipeline
from src.utils.helpers import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

config = load_config()
rag_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """Carga el pipeline de forma lazy (una sola vez)."""
    global rag_pipeline
    if rag_pipeline is None:
        logger.info("Cargando RAG Pipeline...")
        rag_pipeline = RAGPipeline(config)
    return rag_pipeline


def respond(message: str, history: list[list[str]]) -> str:
    """
    Función principal del chatbot.
    Compatible con gr.ChatInterface (formato messages).
    """
    if not message.strip():
        return "Por favor, escribe una pregunta sobre iTimeControl."

    try:
        pipeline = get_pipeline()
        result = pipeline.generate(message)
        answer = result["answer"]

        if result["sources"]:
            sources_str = "\n".join(
                f"• {s}" for s in result["sources"] if s
            )
            answer += f"\n\n---\n📄 **Fuentes consultadas:**\n{sources_str}"

        return answer

    except FileNotFoundError as e:
        return (
            f"⚠️ El sistema aún no está configurado: {e}\n\n"
            "Ejecuta el pipeline completo de preprocesamiento y entrenamiento primero."
        )
    except Exception as e:
        logger.error(f"Error en chatbot: {e}")
        return f"❌ Error interno: {str(e)}"


DESCRIPTION = """
# 🤖 Asistente iTimeControl

Bienvenido al asistente conversacional inteligente para el sistema **iTimeControl**.

Puedo ayudarte con consultas operativas sobre:
- Registro de asistencia y marcaciones
- Gestión de empleados y turnos
- Reportes y exportación de datos
- Configuración del sistema
- Permisos, vacaciones y solicitudes

---
*Tesis: Asistente conversacional basado en fine-tuning + RAG*
"""

EXAMPLES = [
    "¿Cómo registro la asistencia de un empleado?",
    "¿Cómo genero un reporte de horas trabajadas?",
    "¿Cómo asigno un turno a un empleado?",
    "¿Cómo configuro los horarios de trabajo?",
    "¿Cómo exporto los datos de asistencia?",
]


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="iTimeControl Assistant",
        theme=gr.themes.Soft(primary_hue="blue"),
    ) as demo:
        gr.Markdown(DESCRIPTION)

        gr.ChatInterface(
            fn=respond,
            examples=EXAMPLES,
            retry_btn="🔄 Reintentar",
            undo_btn="↩️ Deshacer",
            clear_btn="🗑️ Limpiar chat",
            chatbot=gr.Chatbot(
                height=500,
                label="Chat con iTimeControl Assistant",
                show_label=True,
            ),
            textbox=gr.Textbox(
                placeholder="Escribe tu pregunta sobre iTimeControl...",
                container=False,
                scale=7,
            ),
        )

        with gr.Accordion("ℹ️ Sobre el sistema", open=False):
            gr.Markdown(
                "Este asistente utiliza **Fine-Tuning** (LoRA/QLoRA) y "
                "**RAG** (Retrieval-Augmented Generation) para responder "
                "preguntas operativas sobre el sistema iTimeControl."
            )

    return demo


def main():
    demo = build_ui()
    api_cfg = config.get("api", {})
    demo.launch(
        server_name=api_cfg.get("host", "0.0.0.0"),
        server_port=api_cfg.get("port", 7860),
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
