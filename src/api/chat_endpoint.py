"""
api/chat_endpoint.py
Endpoint REST con FastAPI para el asistente iTimeControl.

Uso:
    uvicorn src.api.chat_endpoint:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    POST /chat   — genera una respuesta
    GET  /health — verifica estado del servicio
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.rag.pipeline import RAGPipeline
from src.utils.helpers import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Estado global del pipeline (cargado al inicio)
_pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el pipeline al arrancar y lo libera al cerrar."""
    global _pipeline
    logger.info("Inicializando RAG Pipeline...")
    config = load_config()
    _pipeline = RAGPipeline(config)
    logger.info("Pipeline listo. API en línea.")
    yield
    logger.info("Cerrando API.")


app = FastAPI(
    title="iTimeControl Assistant API",
    description="Asistente conversacional inteligente para el sistema iTimeControl",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ──────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000,
                          description="Pregunta sobre el sistema iTimeControl")

class ChatResponse(BaseModel):
    answer:     str
    sources:    list[str]
    num_chunks: int


class HealthResponse(BaseModel):
    status:  str
    model:   str
    indexed: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
async def health():
    """Verifica el estado del servicio."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline no inicializado")

    config = load_config()
    return HealthResponse(
        status="ok",
        model=config["model"]["base_model"],
        indexed=_pipeline.retriever.index.ntotal,
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Genera una respuesta a una pregunta sobre iTimeControl.

    El pipeline recupera contexto relevante desde los documentos
    indexados y lo usa para generar una respuesta precisa.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline no inicializado")

    logger.info(f"Pregunta recibida: {request.question[:80]}")

    try:
        result = _pipeline.generate(request.question)
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            num_chunks=result["num_chunks"],
        )
    except Exception as e:
        logger.error(f"Error generando respuesta: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")
