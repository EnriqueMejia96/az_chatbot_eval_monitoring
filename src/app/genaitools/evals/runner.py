from __future__ import annotations

import os
import inspect
from typing import Any, Dict, Iterable, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.genaitools.config import settings
from azure.identity import DefaultAzureCredential

# https://learn.microsoft.com/en-us/python/api/azure-ai-evaluation/azure.ai.evaluation
from azure.ai.evaluation import (
    AzureOpenAIModelConfiguration,
    # Performance & quality (NLP)
    F1ScoreEvaluator,
    RougeScoreEvaluator,
    GleuScoreEvaluator,
    BleuScoreEvaluator,
    MeteorScoreEvaluator,
    # Performance & quality (AI-assisted)
    GroundednessEvaluator,
    RelevanceEvaluator,
    CoherenceEvaluator,
    FluencyEvaluator,
    SimilarityEvaluator,
    RetrievalEvaluator,
    # Risk & safety (AI-assisted)
    ViolenceEvaluator,
    SexualEvaluator,
    SelfHarmEvaluator,
    HateUnfairnessEvaluator,
    IndirectAttackEvaluator,
    # ProtectedMaterialEvaluator, Not support for gen_ai.evaluator.id
    # Composite
    QAEvaluator,
    ContentSafetyEvaluator,
)

# --------------------------------------------------------------------------------------
# Judge model configuration (used by evaluators that accept model_config)
# --------------------------------------------------------------------------------------
JUDGE_ENDPOINT: Optional[str] = os.getenv("AZURE_ENDPOINT")
JUDGE_API_KEY: Optional[str] = os.getenv("AZURE_API_KEY")
JUDGE_DEPLOYMENT: Optional[str] = os.getenv("AZURE_DEPLOYMENT_NAME")
JUDGE_API_VERSION: str = os.getenv("AZURE_API_VERSION", "2024-10-21")

_model_config: Optional[AzureOpenAIModelConfiguration] = None
if JUDGE_ENDPOINT and JUDGE_API_KEY and JUDGE_DEPLOYMENT:
    _model_config = AzureOpenAIModelConfiguration(
        azure_endpoint=JUDGE_ENDPOINT,
        api_key=JUDGE_API_KEY,
        azure_deployment=JUDGE_DEPLOYMENT,
        api_version=JUDGE_API_VERSION,
    )

# --------------------------------------------------------------------------------------
# Azure AI Project info for safety evaluators (require ctor args)
# --------------------------------------------------------------------------------------
AZ_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
AZ_RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP_NAME")
AZ_PROJECT_NAME = os.getenv("AZURE_PROJECT_NAME")
AZ_PROJECT_URL = os.getenv("PROJECT_ENDPOINT")

def _normalize(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")

EVAL_REGISTRY: Dict[str, Any] = {
    # Quality (NLP)
    "f1": F1ScoreEvaluator,
    "rouge": RougeScoreEvaluator,
    "gleu": GleuScoreEvaluator,
    "bleu": BleuScoreEvaluator,
    "meteor": MeteorScoreEvaluator,

    # Quality (AI-assisted)
    "groundedness": GroundednessEvaluator,
    "relevance": RelevanceEvaluator,
    "coherence": CoherenceEvaluator,
    "fluency": FluencyEvaluator,
    "similarity": SimilarityEvaluator,
    "retrieval": RetrievalEvaluator,

    # Safety
    "violence": ViolenceEvaluator,
    "sexual": SexualEvaluator,
    "self_harm": SelfHarmEvaluator,
    "hate_unfairness": HateUnfairnessEvaluator,
    "indirect_attack": IndirectAttackEvaluator,
    #"protected_material": ProtectedMaterialEvaluator,

    # Composite
    "qa": QAEvaluator,
    "content_safety": ContentSafetyEvaluator,
}

# Cache evaluators that don't depend on secrets/credentials
_EVAL_CACHE: Dict[tuple[str, Optional[int]], Any] = {}

def _resolve_project_and_credential(ctor_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Build ctor kwargs ('azure_ai_project', 'credential') for safety evaluators.
    Priority:
      1) values in ctor_overrides
      2) AZURE_AI_PROJECT_URL (string form)
      3) subscription/resource_group/project dict form
    """
    ctor_overrides = ctor_overrides or {}
    out: Dict[str, Any] = {}

    # azure_ai_project
    if "azure_ai_project" in ctor_overrides:
        out["azure_ai_project"] = ctor_overrides["azure_ai_project"]
    else:
        if AZ_PROJECT_URL:
            out["azure_ai_project"] = AZ_PROJECT_URL
        elif AZ_SUBSCRIPTION_ID and AZ_RESOURCE_GROUP and AZ_PROJECT_NAME:
            out["azure_ai_project"] = {
                "subscription_id": AZ_SUBSCRIPTION_ID,
                "resource_group_name": AZ_RESOURCE_GROUP,
                "project_name": AZ_PROJECT_NAME,
            }

    # credential
    if "credential" in ctor_overrides:
        out["credential"] = ctor_overrides["credential"]
    else:
        if "azure_ai_project" in out:
            out["credential"] = DefaultAzureCredential()

    return out

def _build_kwargs_for_constructor(
    cls: Any,
    threshold: Optional[int],
    model_config: Optional[AzureOpenAIModelConfiguration],
    ctor_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Only pass constructor args that the evaluator supports.
    Also inject azure_ai_project/credential when required & available.
    """
    try:
        sig = inspect.signature(cls)
    except (TypeError, ValueError):
        sig = None

    ctor_kwargs: Dict[str, Any] = {}
    if sig:
        # model_config / threshold
        if "model_config" in sig.parameters and model_config is not None:
            ctor_kwargs["model_config"] = model_config
        if "threshold" in sig.parameters and threshold is not None:
            ctor_kwargs["threshold"] = threshold

        # project/credential (for safety evaluators and some composites)
        needs_project = "azure_ai_project" in sig.parameters
        needs_cred = "credential" in sig.parameters
        if needs_project or needs_cred:
            resolved = _resolve_project_and_credential(ctor_overrides)
            if needs_project and "azure_ai_project" in resolved:
                ctor_kwargs["azure_ai_project"] = resolved["azure_ai_project"]
            if needs_cred and "credential" in resolved:
                ctor_kwargs["credential"] = resolved["credential"]

    return ctor_kwargs

def _get_evaluator(
    name: str,
    *,
    threshold: Optional[int] = None,
    ctor_overrides: Optional[Dict[str, Any]] = None,
) -> Any:
    key = _normalize(name)
    if key not in EVAL_REGISTRY:
        raise ValueError(f"Unknown evaluator '{name}'. Known: {', '.join(sorted(EVAL_REGISTRY))}")

    # Use settings.eval_threshold when caller doesn't override
    used_threshold = threshold if threshold is not None else settings.eval_threshold

    cls = EVAL_REGISTRY[key]
    ctor_kwargs = _build_kwargs_for_constructor(cls, used_threshold, _model_config, ctor_overrides)

    # If the ctor includes credential/project, skip cache to avoid sharing sensitive instances
    uses_secret_bits = any(k in ctor_kwargs for k in ("credential", "azure_ai_project"))
    cache_key = (key, used_threshold) if not uses_secret_bits else None

    if cache_key and cache_key in _EVAL_CACHE:
        return _EVAL_CACHE[cache_key]

    inst = cls(**ctor_kwargs)

    if cache_key:
        _EVAL_CACHE[cache_key] = inst
    return inst

def _filter_kwargs_for_call(ev: Any, **kwargs) -> Dict[str, Any]:
    """
    Keep only kwargs that the evaluator's __call__ accepts.
    If the signature is variadic (*args/**kwargs) or not introspectable,
    pass through all non-None kwargs.
    """
    fn = getattr(ev, "__call__", ev)
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        # Can't introspect → pass everything non-None
        return {k: v for k, v in kwargs.items() if v is not None}

    params = list(sig.parameters.values())
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
    has_var_pos = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)

    # If __call__ is variadic, don't filter at all
    if has_var_kw or has_var_pos:
        return {k: v for k, v in kwargs.items() if v is not None}

    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed and v is not None}

    # If filtering removed everything but we *do* have inputs, fall back to pass-through
    if not filtered and any(v is not None for v in kwargs.values()):
        return {k: v for k, v in kwargs.items() if v is not None}

    # Required args check (only for non-variadic signatures)
    required = {
        p.name
        for p in params
        if p.default is inspect._empty
        and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    missing = [r for r in required if r not in filtered]
    if missing:
        raise ValueError(
            f"Evaluator '{ev.__class__.__name__}' requires {missing}; provided keys: {sorted(filtered.keys())}"
        )

    return filtered

# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
def run_evaluator(
    evaluator: str,
    *,
    response: Optional[str] = None,
    query: Optional[str] = None,
    context: Optional[str] = None,
    ground_truth: Optional[str] = None,
    references: Optional[list[str] | str] = None,
    retrieved_contexts: Optional[list[str]] = None,  # for RetrievalEvaluator
    threshold: Optional[int] = None,
    ctor_overrides: Optional[Dict[str, Any]] = None,  # ctor-level overrides like azure_ai_project/credential
    **extra_kwargs: Any,
) -> Dict[str, Any]:
    key = _normalize(evaluator)
    if key == "qa" and (not settings.eval_qa_enable or not ground_truth):
        return {"__skipped__": True, "__reason__": "QA disabled or missing ground_truth"}

    ev = _get_evaluator(evaluator, threshold=threshold, ctor_overrides=ctor_overrides)
    call_kwargs = _filter_kwargs_for_call(
        ev,
        response=response,
        query=query,
        context=context,
        ground_truth=ground_truth,
        references=references,
        retrieved_contexts=retrieved_contexts,
        **extra_kwargs,
    )
    return ev(**call_kwargs)

def run_evaluators(
    evaluators: Union[str, Iterable[str]],
    *,
    response: Optional[str] = None,
    query: Optional[str] = None,
    context: Optional[str] = None,
    ground_truth: Optional[str] = None,
    references: Optional[list[str] | str] = None,
    retrieved_contexts: Optional[list[str]] = None,
    threshold: Optional[int] = None,
    per_eval_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,  # call-time overrides per evaluator
    per_eval_ctor: Optional[Dict[str, Dict[str, Any]]] = None,    # ctor-time overrides per evaluator
    ctor_overrides: Optional[Dict[str, Any]] = None,              # global ctor overrides
    parallel: bool = False,
    max_workers: int = 4,
    quiet: bool = True,
) -> Dict[str, Dict[str, Any]]:
    if isinstance(evaluators, str):
        evaluators = [evaluators]
    evaluators = list(evaluators)

    common = dict(
        response=response,
        query=query,
        context=context,
        ground_truth=ground_truth,
        references=references,
        retrieved_contexts=retrieved_contexts,
        threshold=threshold,
    )
    per_eval_kwargs = per_eval_kwargs or {}
    per_eval_ctor = per_eval_ctor or {}

    def _one(raw_name: str) -> tuple[str, Dict[str, Any]]:
        name = _normalize(raw_name)
        try:
            if name == "qa" and (not settings.eval_qa_enable or not common.get("ground_truth")):
                return name, {"__skipped__": True, "__reason__": "QA disabled or missing ground_truth"}

            # Merge ctor overrides (global + per-evaluator)
            merged_ctor: Dict[str, Any] = {}
            if ctor_overrides:
                merged_ctor.update(ctor_overrides)
            if name in per_eval_ctor:
                merged_ctor.update(per_eval_ctor[name])

            ev = _get_evaluator(name, threshold=threshold, ctor_overrides=merged_ctor)

            # Merge call kwargs (common + per-evaluator), remove 'threshold' (ctor-only)
            merged_call = {**common, **(per_eval_kwargs.get(name, {}) or {})}
            merged_call.pop("threshold", None)

            call_kwargs = _filter_kwargs_for_call(ev, **merged_call)
            out = ev(**call_kwargs)
            return name, out
        except Exception as e:
            if quiet:
                return name, {"__error__": str(e)}
            raise

    results: Dict[str, Dict[str, Any]] = {}

    if parallel and len(evaluators) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut_map = {ex.submit(_one, n): n for n in evaluators}
            for fut in as_completed(fut_map):
                name, out = fut.result()
                results[name] = out
    else:
        for n in evaluators:
            name, out = _one(n)
            results[name] = out

    return results

__all__ = ["run_evaluator", "run_evaluators", "EVAL_REGISTRY"]
