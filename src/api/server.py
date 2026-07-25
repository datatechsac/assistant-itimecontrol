"""
api/server.py
API REST (FastAPI) que expone el pipeline RAG para la demo HTML.

Uso:
    python -m src.api.server
    (o) uvicorn src.api.server:app --host 0.0.0.0 --port 8000
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.rag.pipeline import RAGPipeline
from src.utils.helpers import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

config = load_config()
app = FastAPI(title="iTimeControl Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_pipeline: RAGPipeline | None = None
STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"


def get_pipeline() -> RAGPipeline:
    global rag_pipeline
    if rag_pipeline is None:
        logger.info("Cargando RAG Pipeline...")
        rag_pipeline = RAGPipeline(config)
    return rag_pipeline


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    num_chunks: int


@app.post("/api/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    question = payload.question.strip()
    if not question:
        return AskResponse(answer="Por favor, escribe una pregunta sobre iTimeControl.", sources=[], num_chunks=0)

    try:
        pipeline = get_pipeline()
        result = pipeline.generate(question)
        return AskResponse(
            answer=result["answer"],
            sources=result["sources"],
            num_chunks=result["num_chunks"],
        )
    except FileNotFoundError as e:
        return AskResponse(
            answer=f"El sistema aún no está configurado: {e}",
            sources=[],
            num_chunks=0,
        )
    except Exception as e:
        logger.error(f"Error en /api/ask: {e}", exc_info=True)
        return AskResponse(answer=f"Error interno: {e}", sources=[], num_chunks=0)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def main() -> None:
    import uvicorn
    api_cfg = config.get("api", {})
    uvicorn.run(
        "src.api.server:app",
        host=api_cfg.get("host", "0.0.0.0"),
        port=api_cfg.get("port", 8000),
        reload=False,
    )


if __name__ == "__main__":
    main()
