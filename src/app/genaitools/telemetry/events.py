from __future__ import annotations
from typing import Any, Dict, Optional, Literal
from numbers import Number

SAFETY_IDS = {
    "indirect_attack":    "azureai://built-in/evaluators/indirect_attack",
    "violence":           "azureai://built-in/evaluators/violence",
    "sexual":             "azureai://built-in/evaluators/sexual",
    "self_harm":          "azureai://built-in/evaluators/self_harm",
    "hate_unfairness":    "azureai://built-in/evaluators/hate_unfairness",
    "code_vulnerability": "azureai://built-in/evaluators/code_vulnerability",
}

PRIMARY_SCORE_KEYS = {
    # quality
    "coherence": ["coherence", "gpt_coherence"],
    "fluency":   ["fluency", "gpt_fluency"],
    "relevance": ["relevance", "gpt_relevance"],
    "similarity":["similarity"],
    "groundedness": ["groundedness"],
    "bleu":      ["bleu"],
    "meteor":    ["meteor"],
    "gleu":      ["gleu"],
    "f1":        ["f1", "f1_score"],
    "retrieval": ["retrieval", "retrieval_score", "f1_score", "precision", "recall", "similarity"],

    # safety
    "violence":  ["violence", "score", "value"],
    "sexual":    ["sexual", "score", "value"],
    "self_harm": ["self_harm", "score", "value"],
    "hate_unfairness": ["hate_unfairness", "score", "value"],
    "code_vulnerability": ["code_vulnerability", "score", "value"],
    # indirect_attack handled specially
}

REASON_KEYS = {
    "coherence": "coherence_reason",
    "fluency":   "fluency_reason",
    "relevance": "relevance_reason",
    "similarity":"similarity_reason",
    "groundedness": "groundedness_reason",
    "retrieval": "retrieval_reason",
    "violence":  "violence_reason",
    "sexual":    "sexual_reason",
    "self_harm": "self_harm_reason",
    "hate_unfairness": "hate_unfairness_reason",
    "code_vulnerability": "code_vulnerability_reason",
    "indirect_attack": "xpia_reason",
}

def _norm(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")

def _as_float_boolaware(x: Any) -> Optional[float]:
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    if isinstance(x, Number):
        return float(x)
    return None

def _first_numeric(d: Dict[str, Any], keys: list[str]) -> Optional[float]:
    for k in keys:
        if k in d:
            v = _as_float_boolaware(d[k])
            if v is not None:
                return v
    for k in ("score", "value"):
        if k in d:
            v = _as_float_boolaware(d[k])
            if v is not None:
                return v
    for v in d.values():
        fv = _as_float_boolaware(v)
        if fv is not None:
            return fv
    return None

def _score_for_rouge(res: Dict[str, Any]) -> Optional[float]:
    pref = ("rougeL_f1", "rouge1_f1", "rouge2_f1", "rougeLsum_f1")
    s = _first_numeric(res, list(pref))
    if s is not None:
        return s
    f1s = [v for k, v in res.items() if k.startswith("rouge") and k.endswith("_f1") and isinstance(v, Number)]
    return float(sum(f1s) / len(f1s)) if f1s else None

def _extract_indirect_attack(res: Dict[str, Any]) -> tuple[float, Dict[str, Any], Optional[str]]:
    label = res.get("xpia_label")
    if isinstance(label, bool):
        score = 1.0 if label else 0.0
    else:
        subs = [
            res.get("xpia_manipulated_content"),
            res.get("xpia_intrusion"),
            res.get("xpia_information_gathering"),
        ]
        score = 1.0 if any(isinstance(v, bool) and v for v in subs) else 0.0
    extra = {
        "xpia_manipulated_content": 1.0 if res.get("xpia_manipulated_content") else 0.0,
        "xpia_intrusion": 1.0 if res.get("xpia_intrusion") else 0.0,
        "xpia_information_gathering": 1.0 if res.get("xpia_information_gathering") else 0.0,
        "xpia_label": 1.0 if bool(label) else 0.0,
    }
    explanation = res.get("xpia_reason")
    return score, extra, explanation

def emit_eval_event(
    active_span,
    *,
    evaluator: str,                 
    result: Any,                    
    model: str,
    response_id: str,
    numeric_key: Optional[str] = None,  
) -> Optional[float]:
    """
    Auto-detects score + extra attrs from the evaluator result and emits a
    'gen_ai.evaluation.<evaluator>' event that Azure AI Foundry/KQL recognizes.
    Returns the numeric score emitted (or None if no score).
    """
    ev = _norm(evaluator)
    event_name = f"gen_ai.evaluation.{ev}"
    is_safety = ev in SAFETY_IDS
    evaluator_id = SAFETY_IDS.get(ev)

    score: Optional[float] = None
    explanation: Optional[str] = None
    extra_attrs: Dict[str, Any] = {}

    if isinstance(result, dict):
        if ev == "rouge":
            score = _score_for_rouge(result)
        elif ev == "indirect_attack":
            score, extra_attrs, explanation = _extract_indirect_attack(result)
        else:
            if numeric_key:
                score = _first_numeric(result, [numeric_key])
            if score is None:
                score = _first_numeric(result, PRIMARY_SCORE_KEYS.get(ev, []))
            rk = REASON_KEYS.get(ev)
            if rk:
                explanation = result.get(rk)
    else:
        score = _as_float_boolaware(result)

    if score is None:
        try:
            active_span.add_event("gen_ai.evaluation.missing_score", attributes={
                "gen_ai.evaluator.name": ev,
                "gen_ai.request.model": model,
                "gen_ai.response.id": response_id,
            })
        except Exception:
            pass
        return None

    attrs: Dict[str, Any] = {
        "gen_ai.evaluator.name": ev,
        "gen_ai.evaluation.score": float(score),
        "gen_ai.request.model": model,
        "gen_ai.response.id": response_id,
        "gen_ai.evaluation.category": "safety" if is_safety else "quality",
    }
    if evaluator_id:
        attrs["gen_ai.evaluator.id"] = evaluator_id
    if explanation:
        attrs["gen_ai.evaluation.explanation"] = str(explanation)
    for k, v in (extra_attrs or {}).items():
        if v is not None:
            attrs[f"gen_ai.evaluation.{ev}.{k}"] = v

    try:
        active_span.add_event(event_name, attributes=attrs)
    except Exception:
        pass

    return float(score)


Operation = Literal["chat", "embedding", "rag"]

def set_genai_span_attrs(
    span,
    *,
    session_id: Optional[str] = None,
    service: str = "openai",
    operation: Operation = "chat",
    model: Optional[str] = None,
    server_host: str = "",
    temperature: Optional[float] = None,
    embedding_dimension: Optional[int] = None,
    input_tokens: Optional[float] = None,
    output_tokens: Optional[float] = None,
    total_tokens: Optional[float] = None,
    latency_ms: Optional[int] = None,
    n_chunks: Optional[int] = None
) -> None:
    if session_id:
        span.set_attribute("session.id", session_id)
    span.set_attribute("gen_ai.service", service)
    span.set_attribute("gen_ai.operation.name", operation)
    span.set_attribute("server.address", server_host or "")

    if model is not None:
        try:
            span.set_attribute("gen_ai.operation.model", model)
        except (TypeError, ValueError):
            pass

    if latency_ms is not None:
        try:
            span.set_attribute("gen_ai.response.latency_ms", int(latency_ms))
        except (TypeError, ValueError):
            pass
    if input_tokens is not None:
        try:
            span.set_attribute("gen_ai.usage.input_tokens", int(input_tokens))
        except (TypeError, ValueError):
            pass
    if output_tokens is not None:
        try:
            span.set_attribute("gen_ai.usage.output_tokens", int(output_tokens))
        except (TypeError, ValueError):
            pass
    if total_tokens is not None:
        try:
            span.set_attribute("gen_ai.usage.total_tokens", int(total_tokens))
        except (TypeError, ValueError):
            pass

    # Op-specific
    op = (operation or "").lower()
    if op == "chat":
        if temperature is not None:
            try:
                span.set_attribute("gen_ai.request.temperature", float(temperature))
            except (TypeError, ValueError):
                pass
    elif op == "embedding":
        if embedding_dimension is not None:
            try:
                span.set_attribute("gen_ai.operation.dimension", int(embedding_dimension))
            except (TypeError, ValueError):
                pass
    elif op == "rag":
        if n_chunks is not None:
            try:
                span.set_attribute("gen_ai.operation.n_chunks", int(n_chunks))
            except (TypeError, ValueError):
                pass