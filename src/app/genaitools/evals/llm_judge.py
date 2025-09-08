from app.genaitools.config import settings
_HAS_LLM_EVAL = False

try:
    from azure.ai.evaluation import (
        AzureOpenAIModelConfiguration,
        CoherenceEvaluator,
        FluencyEvaluator,
        QAEvaluator,
    )

    JUDGE_ENDPOINT   = os.getenv("AZURE_ENDPOINT")
    JUDGE_API_KEY    = os.getenv("AZURE_API_KEY")
    JUDGE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT_NAME")
    JUDGE_API_VERSION= os.getenv("AZURE_API_VERSION", "2024-10-21")

    if JUDGE_ENDPOINT and JUDGE_API_KEY and JUDGE_DEPLOYMENT:
        _model_config = AzureOpenAIModelConfiguration(
            azure_endpoint=JUDGE_ENDPOINT,
            api_key=JUDGE_API_KEY,
            azure_deployment=JUDGE_DEPLOYMENT,
            api_version=JUDGE_API_VERSION,
        )
        _coherence_eval = CoherenceEvaluator(model_config=_model_config, threshold=settings.eval_threshold)
        _fluency_eval   = FluencyEvaluator(model_config=_model_config,   threshold=settings.eval_threshold)
        _HAS_LLM_EVAL = True
    else:
        _model_config = _coherence_eval = _fluency_eval = None
except Exception:
    _model_config = _coherence_eval = _fluency_eval = None
    _HAS_LLM_EVAL = False

def llm_eval(question_text: str, answer_text: str, context_text: str, ground_truth: str | None = None):
    results = {}
    if not _HAS_LLM_EVAL:
        return results
    try:
        coh = _coherence_eval(query=question_text, response=answer_text)
        results["ai.eval.coherence"]      = float(coh.get("coherence")) if "coherence" in coh else None
        results["ai.eval.coherence_pass"] = 1.0 if str(coh.get("coherence_result","")).lower() == "pass" else 0.0
        if "coherence_reason" in coh:
            results["ai.eval.coherence_reason"] = str(coh["coherence_reason"])
    except Exception:
        pass

    try:
        flu = _fluency_eval(response=answer_text)
        results["ai.eval.fluency"]      = float(flu.get("fluency")) if "fluency" in flu else None
        results["ai.eval.fluency_pass"] = 1.0 if str(flu.get("fluency_result","")).lower() == "pass" else 0.0
        if "fluency_reason" in flu:
            results["ai.eval.fluency_reason"] = str(flu["fluency_reason"])
    except Exception:
        pass

    if settings.eval_qa_enable and ground_truth:
        try:
            qa_eval = QAEvaluator(model_config=_model_config, threshold=settings.eval_threshold)
            qa = qa_eval(query=question_text, context=context_text or "", response=answer_text, ground_truth=ground_truth)
            for k in ("f1_score","similarity","fluency","relevance","coherence","groundedness"):
                if k in qa and qa[k] is not None:
                    results[f"ai.eval.qa.{k}"] = float(qa[k])
            for rk in ("f1_result","similarity_result","fluency_result","relevance_result","coherence_result","groundedness_result"):
                if rk in qa:
                    results[f"ai.eval.qa.{rk.replace('_result','_pass')}"] = 1.0 if str(qa[rk]).lower()=="pass" else 0.0
            for rk in ("fluency_reason","relevance_reason","coherence_reason","groundedness_reason"):
                if rk in qa:
                    results[f"ai.eval.qa.{rk}"] = str(qa[rk])
        except Exception:
            pass
    return {k:v for k,v in results.items() if v is not None}
