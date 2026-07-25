"""
evaluation/llm_judge.py
Evalúa la calidad de las respuestas usando un LLM como juez (LLM-as-judge).

Métricas:
  - faithfulness: ¿la respuesta está fundamentada en el contexto recuperado? (solo RAG)
  - relevancy:    ¿la respuesta atiende directamente la pregunta?
  - correctness:  ¿la respuesta coincide en contenido con la respuesta de referencia?

Uso típico:
    from src.evaluation.llm_judge import judge_batch
    result = judge_batch(items, mode="rag", judge_provider="groq")
"""
import json
import os
import re

from src.utils.logger import get_logger

logger = get_logger(__name__)

RAG_JUDGE_PROMPT = """Eres un evaluador experto e imparcial de sistemas RAG (Retrieval-Augmented Generation).

PREGUNTA:
{question}

CONTEXTO RECUPERADO:
{context}

RESPUESTA DE REFERENCIA (ground truth):
{reference}

RESPUESTA GENERADA POR EL SISTEMA:
{answer}

Evalúa tres criterios, cada uno en escala de 0.0 a 1.0:

1. faithfulness: ¿Todas las afirmaciones de la RESPUESTA GENERADA están respaldadas por el CONTEXTO? (1.0 = totalmente fundamentada; 0.0 = inventa o contradice el contexto)
2. relevancy: ¿La RESPUESTA GENERADA atiende directamente la PREGUNTA? (1.0 = responde completamente; 0.0 = no responde)
3. correctness: ¿El contenido de la RESPUESTA GENERADA coincide con la RESPUESTA DE REFERENCIA, aunque use otras palabras? (1.0 = misma información; 0.0 = información distinta o incorrecta)

Responde ÚNICAMENTE con un JSON válido, sin texto adicional ni markdown, con este formato exacto:
{{"faithfulness": <float>, "relevancy": <float>, "correctness": <float>, "justification": "<explicación breve en una frase>"}}
"""

BASELINE_JUDGE_PROMPT = """Eres un evaluador experto e imparcial de asistentes conversacionales.

PREGUNTA:
{question}

RESPUESTA DE REFERENCIA (ground truth):
{reference}

RESPUESTA GENERADA POR EL SISTEMA (sin acceso a documentos, solo su conocimiento general):
{answer}

Evalúa dos criterios, cada uno en escala de 0.0 a 1.0:

1. relevancy: ¿La RESPUESTA GENERADA atiende directamente la PREGUNTA? (1.0 = responde completamente; 0.0 = no responde o se declara sin información)
2. correctness: ¿El contenido de la RESPUESTA GENERADA coincide con la RESPUESTA DE REFERENCIA, aunque use otras palabras? (1.0 = misma información; 0.0 = información distinta, inventada o incorrecta)

Responde ÚNICAMENTE con un JSON válido, sin texto adicional ni markdown, con este formato exacto:
{{"relevancy": <float>, "correctness": <float>, "justification": "<explicación breve en una frase>"}}
"""

_DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-2.0-flash",
    "anthropic": "claude-haiku-4-5-20251001",
}


def _build_judge_client(provider: str):
    if provider == "groq":
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY no encontrada para el juez LLM.")
        return Groq(api_key=api_key)

    if provider == "gemini":
        import google.generativeai as genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY no encontrada para el juez LLM.")
        genai.configure(api_key=api_key)
        return genai

    if provider == "anthropic":
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY no encontrada para el juez LLM.")
        return anthropic.Anthropic(api_key=api_key)

    raise ValueError(f"Proveedor de juez no soportado: {provider}")


def _call_judge(provider: str, client, model: str, prompt: str) -> str:
    if provider == "groq":
        resp = client.chat.completions.create(
            model=model,
            max_tokens=250,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()

    if provider == "gemini":
        gm = client.GenerativeModel(model)
        resp = gm.generate_content(prompt)
        return resp.text.strip()

    if provider == "anthropic":
        resp = client.messages.create(
            model=model,
            max_tokens=250,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()

    raise ValueError(provider)


def _parse_judge_output(text: str, keys: list[str]) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        logger.warning(f"Salida del juez no parseable: {text[:120]}")
        return {k: 0.0 for k in keys if k != "justification"} | {"justification": "parse_error"}
    try:
        data = json.loads(match.group(0))
        result = {k: float(data.get(k, 0.0)) for k in keys if k != "justification"}
        result["justification"] = str(data.get("justification", ""))
        return result
    except (json.JSONDecodeError, TypeError, ValueError):
        logger.warning(f"Error parseando JSON del juez: {text[:120]}")
        return {k: 0.0 for k in keys if k != "justification"} | {"justification": "parse_error"}


def judge_batch(
    items: list[dict],
    mode: str = "rag",
    judge_provider: str = "groq",
    judge_model: str | None = None,
) -> dict:
    """
    Evalúa un batch de resultados usando un LLM como juez.

    Args:
        items: lista de dicts.
            modo "rag":      requiere 'question', 'context', 'reference', 'answer'
            modo "baseline": requiere 'question', 'reference', 'answer'
        mode: "rag" o "baseline"
        judge_provider: "groq" | "gemini" | "anthropic"
        judge_model: nombre del modelo (usa default del proveedor si es None)

    Returns:
        Dict con métricas promedio + detalle por ítem.
    """
    client = _build_judge_client(judge_provider)
    model = judge_model or _DEFAULT_MODELS[judge_provider]

    if mode == "rag":
        prompt_template = RAG_JUDGE_PROMPT
        keys = ["faithfulness", "relevancy", "correctness", "justification"]
    elif mode == "baseline":
        prompt_template = BASELINE_JUDGE_PROMPT
        keys = ["relevancy", "correctness", "justification"]
    else:
        raise ValueError(f"Modo no soportado: {mode} (usa 'rag' o 'baseline')")

    results = []
    for i, item in enumerate(items, start=1):
        logger.info(f"[judge {mode} {i}/{len(items)}] {item['question'][:60]}...")
        prompt = prompt_template.format(
            question=item["question"],
            context=item.get("context", "") or "(sin contexto)",
            reference=item["reference"],
            answer=item["answer"],
        )
        raw = _call_judge(judge_provider, client, model, prompt)
        verdict = _parse_judge_output(raw, keys)
        results.append({**item, **verdict})

    avg = {}
    score_keys = [k for k in keys if k != "justification"]
    for k in score_keys:
        avg[k] = round(sum(r[k] for r in results) / len(results), 4)

    logger.info(f"LLM-judge ({mode}) promedio: {avg}")

    return {
        **avg,
        "mode": mode,
        "judge_provider": judge_provider,
        "judge_model": model,
        "details": results,
    }
