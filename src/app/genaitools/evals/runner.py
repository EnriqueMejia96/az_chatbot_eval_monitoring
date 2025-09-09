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


def set_eval_env_vars(
    *,
    # Judge model configuration
    judge_endpoint: Optional[str] = None,
    judge_api_key: Optional[str] = None,
    judge_deployment: Optional[str] = None,
    judge_api_version: Optional[str] = None,
    # Azure AI Project info
    subscription_id: Optional[str] = None,
    resource_group: Optional[str] = None,
    project_name: Optional[str] = None,
    project_url: Optional[str] = None,
) -> None:

    if judge_endpoint:
        os.environ["AZURE_JUDGE_ENDPOINT"] = judge_endpoint
    if judge_api_key:
        os.environ["AZURE_JUDGE_API_KEY"] = judge_api_key
    if judge_deployment:
        os.environ["AZURE_JUDGE_DEPLOYMENT_NAME"] = judge_deployment
    if judge_api_version:
        os.environ["AZURE_JUDGE_API_VERSION"] = judge_api_version

    if subscription_id:
        os.environ["AZURE_SUBSCRIPTION_ID"] = subscription_id
    if resource_group:
        os.environ["AZURE_RESOURCE_GROUP_NAME"] = resource_group
    if project_name:
        os.environ["AZURE_PROJECT_NAME"] = project_name
    if project_url:
        os.environ["PROJECT_ENDPOINT"] = project_url

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

_EVAL_CACHE: Dict[tuple[str, Optional[int]], Any] = {}

def _resolve_project_and_credential(ctor_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ctor_overrides = ctor_overrides or {}
    out: Dict[str, Any] = {}

    if "azure_ai_project" in ctor_overrides:
        out["azure_ai_project"] = ctor_overrides["azure_ai_project"]
    else:
        if os.getenv("PROJECT_ENDPOINT"):
            out["azure_ai_project"] = os.getenv("PROJECT_ENDPOINT")
        elif os.getenv("AZURE_SUBSCRIPTION_ID") and os.getenv("AZURE_RESOURCE_GROUP_NAME") and os.getenv("AZURE_PROJECT_NAME"):
            out["azure_ai_project"] = {
                "subscription_id": os.getenv("AZURE_SUBSCRIPTION_ID"),
                "resource_group_name": os.getenv("AZURE_RESOURCE_GROUP_NAME"),
                "project_name": os.getenv("AZURE_PROJECT_NAME")
            }

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
    try:
        sig = inspect.signature(cls)
    except (TypeError, ValueError):
        sig = None

    ctor_kwargs: Dict[str, Any] = {}
    if sig:
        if "model_config" in sig.parameters and model_config is not None:
            ctor_kwargs["model_config"] = model_config
        if "threshold" in sig.parameters and threshold is not None:
            ctor_kwargs["threshold"] = threshold

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

    used_threshold = threshold if threshold is not None else settings.eval_threshold

    cls = EVAL_REGISTRY[key]
    _model_config = AzureOpenAIModelConfiguration(
        azure_endpoint=os.getenv("AZURE_JUDGE_ENDPOINT"),
        api_key=os.getenv("AZURE_JUDGE_API_KEY"),
        azure_deployment=os.getenv("AZURE_JUDGE_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_JUDGE_API_VERSION", "2024-10-21"),
    )
    ctor_kwargs = _build_kwargs_for_constructor(cls, used_threshold, _model_config, ctor_overrides)

    uses_secret_bits = any(k in ctor_kwargs for k in ("credential", "azure_ai_project"))
    cache_key = (key, used_threshold) if not uses_secret_bits else None

    if cache_key and cache_key in _EVAL_CACHE:
        return _EVAL_CACHE[cache_key]

    inst = cls(**ctor_kwargs)

    if cache_key:
        _EVAL_CACHE[cache_key] = inst
    return inst

def _filter_kwargs_for_call(ev: Any, **kwargs) -> Dict[str, Any]:

    fn = getattr(ev, "__call__", ev)
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return {k: v for k, v in kwargs.items() if v is not None}

    params = list(sig.parameters.values())
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
    has_var_pos = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)

    if has_var_kw or has_var_pos:
        return {k: v for k, v in kwargs.items() if v is not None}

    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed and v is not None}

    if not filtered and any(v is not None for v in kwargs.values()):
        return {k: v for k, v in kwargs.items() if v is not None}

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


def run_evaluator(
    evaluator: str,
    *,
    response: Optional[str] = None,
    query: Optional[str] = None,
    context: Optional[str] = None,
    ground_truth: Optional[str] = None,
    references: Optional[list[str] | str] = None,
    retrieved_contexts: Optional[list[str]] = None,
    threshold: Optional[int] = None,
    ctor_overrides: Optional[Dict[str, Any]] = None,
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
    per_eval_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    per_eval_ctor: Optional[Dict[str, Dict[str, Any]]] = None,
    ctor_overrides: Optional[Dict[str, Any]] = None,
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

            merged_ctor: Dict[str, Any] = {}
            if ctor_overrides:
                merged_ctor.update(ctor_overrides)
            if name in per_eval_ctor:
                merged_ctor.update(per_eval_ctor[name])

            ev = _get_evaluator(name, threshold=threshold, ctor_overrides=merged_ctor)

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
