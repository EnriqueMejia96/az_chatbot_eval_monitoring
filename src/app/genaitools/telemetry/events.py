def emit_eval_event(active_span, evaluator: str, score: float, model: str, response_id: str):
    name = f"gen_ai.evaluation.{evaluator}"
    attrs = {
        "event.name": name,
        "gen_ai.evaluation.score": float(score),
        "gen_ai.evaluator.name": evaluator,
        "gen_ai.request.model": model,
        "gen_ai.response.id": response_id,
    }
    try:
        active_span.add_event(name, attributes=attrs)
    except Exception:
        pass
