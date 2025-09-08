import os, inspect
import azure.ai.evaluation as _eval_mod
from app.genaitools.config import settings
from .utils import safe_call_eval, extract_numeric

_SAFETY_CLASS_BY_KEY = {
    "code_vulnerability": "CodeVulnerabilityEvaluator",
    "hate_speechness": "HateUtterancesEvaluator",
    "indirect_attack": "IndirectAttackEvaluator",
    "self_harm": "SelfHarmEvaluator",
    "sexual": "SexualEvaluator",
    "violence": "ViolenceEvaluator",
}
_SAFETY_ID_BY_KEY = {
    "code_vulnerability": "azureai://built-in/evaluators/code_vulnerability",
    "hate_speechness": "azureai://built-in/evaluators/hate_utterances",
    "indirect_attack": "azureai://built-in/evaluators/indirect_attack",
    "self_harm": "azureai://built-in/evaluators/self_harm",
    "sexual": "azureai://built-in/evaluators/sexual",
    "violence": "azureai://built-in/evaluators/violence",
}

_safety_instances = {}
_HAS_SAFETY = False

try:
    # reuse judge model if present, otherwise evaluators run with defaults
    for key, alias_name in _SAFETY_CLASS_BY_KEY.items():
        Ev = getattr(_eval_mod, alias_name, None)
        if not Ev:
            _safety_instances[key] = None
            continue
        try:
            kwargs = {}
            if "threshold" in inspect.signature(Ev).parameters:
                kwargs["threshold"] = settings.eval_threshold
            _safety_instances[key] = Ev(**kwargs)
        except Exception:
            _safety_instances[key] = None
    _HAS_SAFETY = any(v is not None for v in _safety_instances.values())
except Exception:
    _safety_instances = {}
    _HAS_SAFETY = False

def safety_eval(question_text: str, answer_text: str, context_text: str) -> dict:
    if not (_HAS_SAFETY and settings.eval_safety_enable):
        return {}
    results = {}
    for key, ev in _safety_instances.items():
        if not ev:
            continue
        try:
            out = safe_call_eval(ev, query=question_text, response=answer_text, context=context_text or "")
            score = extract_numeric(out, key)
            if score is not None:
                results[f"ai.eval.safety.{key}"] = float(score)
        except Exception:
            continue
    return results

SAFETY_IDS = _SAFETY_ID_BY_KEY  # for lookup when emitting events
HAS_SAFETY = _HAS_SAFETY
