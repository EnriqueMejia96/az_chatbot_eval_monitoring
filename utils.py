import os
import uuid
import time
import re
import numpy as np
from dotenv import load_dotenv
from urllib.parse import urlparse

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor

# ---------- Load env ----------
load_dotenv()
project_endpoint = os.getenv("PROJECT_ENDPOINT")
embedding_deployment = os.getenv("EMBEDDING_DEPLOYMENT")

EVAL_ATTACH_TO_CHAT = os.getenv("EVAL_ATTACH_TO_CHAT", "true").lower() in ("1", "true", "yes")


if not project_endpoint:
    raise RuntimeError("PROJECT_ENDPOINT must be set as environment variables.")

# Helpful service name + content capture (consider disabling in prod)
os.environ.setdefault("OTEL_SERVICE_NAME", "rag-app")
os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true")

# Toggle extra eval embedding calls (answer vs. context similarity)
EVAL_CONTEXT_SIM = os.getenv("EVAL_CONTEXT_SIMILARITY", "true").lower() in ("1", "true", "yes")
EVAL_THRESHOLD = int(os.getenv("EVAL_THRESHOLD", "3"))
EVAL_QA_ENABLE = os.getenv("EVAL_QA_ENABLE", "false").lower() in ("1", "true", "yes")

# ---------- OpenTelemetry ----------
tracer = trace.get_tracer(__name__)
SESSION_ID = str(uuid.uuid4())

# For trace attributes like server.address
_server_host = urlparse(project_endpoint).hostname or ""

# ---------- Azure AI Project & Observability ----------
credential = DefaultAzureCredential(
    exclude_environment_credential=False,
    exclude_managed_identity_credential=False,
    exclude_shared_token_cache_credential=True,
    exclude_visual_studio_code_credential=True,
    exclude_powershell_credential=True,
    exclude_cli_credential=False,
)

project_client = AIProjectClient(
    credential=credential,
    endpoint=project_endpoint,
)

ai_conn = project_client.telemetry.get_application_insights_connection_string()
if not ai_conn:
    raise RuntimeError("No App Insights connected to this Project – configure it in the portal (Tracing).")
configure_azure_monitor(connection_string=ai_conn)

# Optional: export traces to console (useful in CI)
if os.getenv("OTEL_TRACE_TO_CONSOLE", "false").lower() in ("1", "true", "yes"):
    try:
        from opentelemetry import trace as trace_api
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
        provider = trace_api.get_tracer_provider()
        if hasattr(provider, "add_span_processor"):
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    except Exception:
        pass

# Instrument OpenAI-compatible client
OpenAIInstrumentor().instrument()

# OpenAI-compatible client from the Project
chat_client = project_client.get_openai_client(api_version="2024-10-21")

# ---------- Optional LLM-judge evaluators (Coherence, Fluency, QA) ----------
_HAS_LLM_EVAL = False
try:
    from azure.ai.evaluation import (
        AzureOpenAIModelConfiguration,
        CoherenceEvaluator,
        FluencyEvaluator,
        QAEvaluator,
    )

    # Use the same env var names as the docs (you can also set JUDGE_* variants if you prefer)
    JUDGE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
    JUDGE_API_KEY = os.getenv("AZURE_API_KEY")
    JUDGE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT_NAME")
    JUDGE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-10-21")

    if JUDGE_ENDPOINT and JUDGE_API_KEY and JUDGE_DEPLOYMENT:
        _model_config = AzureOpenAIModelConfiguration(
            azure_endpoint=JUDGE_ENDPOINT,
            api_key=JUDGE_API_KEY,
            azure_deployment=JUDGE_DEPLOYMENT,
            api_version=JUDGE_API_VERSION,
        )
        _coherence_eval = CoherenceEvaluator(model_config=_model_config, threshold=EVAL_THRESHOLD)
        _fluency_eval = FluencyEvaluator(model_config=_model_config, threshold=EVAL_THRESHOLD)
        # QA evaluator will be created lazily only if used
        _HAS_LLM_EVAL = True
    else:
        _model_config = None
        _coherence_eval = None
        _fluency_eval = None
except Exception:
    _model_config = None
    _coherence_eval = None
    _fluency_eval = None
    _HAS_LLM_EVAL = False

# ---------- Your original helpers (same API) ----------
def text_embedding(text=[]):
    # Trace-only: richer span name & attrs; no logic change
    with tracer.start_as_current_span("embeddings.create") as span:
        span.set_attribute("session.id", SESSION_ID)
        span.set_attribute("gen_ai.system", "openai")
        span.set_attribute("gen_ai.operation.name", "embedding")
        span.set_attribute("gen_ai.request.model", embedding_deployment)
        span.set_attribute("server.address", _server_host)
        try:
            t0 = time.time()
            resp = chat_client.embeddings.create(
                model=embedding_deployment,
                input=text,
                encoding_format="float",
            )
            dt_ms = int((time.time() - t0) * 1000)
            span.set_attribute("gen_ai.response.latency_ms", dt_ms)

            emb = resp.data[0].embedding
            span.set_attribute("embedding.dimension", len(emb))

            if hasattr(resp, "id"):
                span.set_attribute("gen_ai.response.id", resp.id)
            if hasattr(resp, "model"):
                span.set_attribute("gen_ai.response.model", resp.model)

            usage = getattr(resp, "usage", None)
            if usage:
                span.set_attribute("gen_ai.usage.input_tokens", getattr(usage, "prompt_tokens", 0) or 0)
                span.set_attribute("gen_ai.usage.total_tokens", getattr(usage, "total_tokens", 0) or 0)
            return emb
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise

def get_dot_product(row):
    return np.dot(row, query_vector)

def cosine_similarity(row):
    denominator1 = np.linalg.norm(row)
    denominator2 = np.linalg.norm(query_vector.ravel())
    dot_prod = np.dot(row, query_vector)
    return dot_prod / (denominator1 * denominator2)

def get_context_from_query(query, vector_store, n_chunks=5):
    # Trace-only: wrap selection in its own span; no logic change
    with tracer.start_as_current_span("rag.select_context") as span:
        span.set_attribute("session.id", SESSION_ID)
        span.set_attribute("rag.top_k", n_chunks)

        global query_vector
        query_vector = np.array(text_embedding(query))
        top_matched = (
            vector_store["Embedding"]
            .apply(cosine_similarity)
            .sort_values(ascending=False)[:n_chunks]
            .index
        )
        top_matched_df = vector_store[vector_store.index.isin(top_matched)][["Chunks"]]
        chunks = list(top_matched_df["Chunks"])
        span.set_attribute("rag.chunks.count", len(chunks))
        return chunks

custom_prompt = """
Eres una Inteligencia Artificial super avanzada que trabaja asistente personal.
Utilice los RESULTADOS DE BÚSQUEDA SEMANTICA para responder las preguntas del usuario. 
Solo debes utilizar la informacion de la BUSQUEDA SEMANTICA si es que hace sentido y tiene relacion con la pregunta del usuario.
Si la respuesta no se encuentra dentro del contexto de la búsqueda semántica, no inventes una respuesta, y responde amablemente que no tienes información para responder.

RESULTADOS DE BÚSQUEDA SEMANTICA:
{source}

Lee cuidadosamente las instrucciones, respira profundo y escribe una respuesta para el usuario!
"""

# ---------- Basic evaluation (numeric attributes show in tracing) ----------
def _last_user_message(messages):
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""

def _extract_context_from_system(messages):
    if not messages:
        return ""
    sys = messages[0].get("content", "") if messages[0].get("role") == "system" else ""
    if not sys:
        return ""
    m = re.search(r"RESULTADOS DE BÚSQUEDA SEM[ÁA]NTICA:\s*(.+?)\n\s*Lee cuidadosamente", sys, flags=re.S)
    return m.group(1).strip() if m else ""

def _basic_eval(answer_text: str, question_text: str, context_text: str):
    words = re.findall(r"\w+", answer_text, flags=re.UNICODE)
    length = len(answer_text)
    sentences = sum(answer_text.count(x) for x in [".", "!", "?"])
    lex_div = (len(set(w.lower() for w in words)) / max(1, len(words))) if words else 0.0

    metrics = {
        "ai.evaluation.answer_length": float(length),
        "ai.evaluation.sentences": float(sentences),
        "ai.evaluation.lexical_diversity": float(lex_div),
    }

    if EVAL_CONTEXT_SIM and context_text:
        try:
            a_vec = np.array(text_embedding(answer_text), dtype=np.float32)
            c_vec = np.array(text_embedding(context_text), dtype=np.float32)
            denom = (np.linalg.norm(a_vec) * np.linalg.norm(c_vec)) or 1.0
            sim = float(np.dot(a_vec, c_vec) / denom)
            metrics["ai.evaluation.context_similarity"] = sim
        except Exception:
            pass

    return metrics

# ---------- LLM-judge evaluation (Coherence / Fluency / optional QA) ----------

def _emit_eval_event(active_span, evaluator: str, score: float, model: str, response_id: str):
    # Emits a trace event that Foundry's KQL will pick up
    name = f"gen_ai.evaluation.{evaluator}"
    attrs = {
        "event.name": name,                         # optional; KQL also falls back to 'message'
        "gen_ai.evaluation.score": float(score),    # REQUIRED for the chart
        "gen_ai.evaluator.name": evaluator,         # optional; KQL derives from event name if missing
        "gen_ai.request.model": model,              # REQUIRED for model filter
        "gen_ai.response.id": response_id,          # REQUIRED for the join with the inference call
    }
    try:
        active_span.add_event(name, attributes=attrs)
    except Exception:
        pass


def _llm_eval(question_text: str, answer_text: str, context_text: str, ground_truth: str | None = None):
    results = {}
    if not _HAS_LLM_EVAL:
        return results

    # Coherence
    try:
        # Coherence accepts (query, response); both help the judge
        coh = _coherence_eval(query=question_text, response=answer_text)
        # Likert scores are numeric (1..5); pass/fail is string
        results["ai.eval.coherence"] = float(coh.get("coherence")) if "coherence" in coh else None
        results["ai.eval.coherence_pass"] = 1.0 if str(coh.get("coherence_result", "")).lower() == "pass" else 0.0
        # reasons as strings (visible in span details)
        if "coherence_reason" in coh:
            results["ai.eval.coherence_reason"] = str(coh["coherence_reason"])
    except Exception:
        pass

    # Fluency
    try:
        flu = _fluency_eval(response=answer_text)
        results["ai.eval.fluency"] = float(flu.get("fluency")) if "fluency" in flu else None
        results["ai.eval.fluency_pass"] = 1.0 if str(flu.get("fluency_result", "")).lower() == "pass" else 0.0
        if "fluency_reason" in flu:
            results["ai.eval.fluency_reason"] = str(flu["fluency_reason"])
    except Exception:
        pass

    # QA (optional; requires ground truth)
    if EVAL_QA_ENABLE and ground_truth:
        try:
            qa_eval = QAEvaluator(model_config=_model_config, threshold=EVAL_THRESHOLD)
            qa = qa_eval(
                query=question_text,
                context=context_text or "",
                response=answer_text,
                ground_truth=ground_truth,
            )
            # numeric fields
            for k in ("f1_score", "similarity", "fluency", "relevance", "coherence", "groundedness"):
                if k in qa and qa[k] is not None:
                    results[f"ai.eval.qa.{k}"] = float(qa[k])
            # pass/fail to 0/1
            for rk in ("f1_result", "similarity_result", "fluency_result", "relevance_result",
                       "coherence_result", "groundedness_result"):
                if rk in qa:
                    results[f"ai.eval.qa.{rk.replace('_result','_pass')}"] = 1.0 if str(qa[rk]).lower() == "pass" else 0.0
            # include one reason if present
            for rk in ("fluency_reason", "relevance_reason", "coherence_reason", "groundedness_reason"):
                if rk in qa:
                    results[f"ai.eval.qa.{rk}"] = str(qa[rk])
        except Exception:
            pass

    return {k: v for k, v in results.items() if v is not None}

def get_response(model, temperature, messages):
    with tracer.start_as_current_span("chat.completions.create") as span:
        span.set_attribute("session.id", SESSION_ID)
        span.set_attribute("gen_ai.system", "openai")
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.request.temperature", float(temperature))
        span.set_attribute("server.address", _server_host)
        try:
            t0 = time.time()
            completion = chat_client.chat.completions.create(
                model=model, temperature=temperature, messages=messages
            )
            dt_ms = int((time.time() - t0) * 1000)
            span.set_attribute("gen_ai.response.latency_ms", dt_ms)

            resp_id  = getattr(completion, "id", "") or ""
            resp_mod = getattr(completion, "model", "") or ""
            if resp_id:  span.set_attribute("gen_ai.response.id", resp_id)
            if resp_mod: span.set_attribute("gen_ai.response.model", resp_mod)

            try:
                fr = completion.choices[0].finish_reason
                if fr is not None:
                    span.set_attribute("gen_ai.response.finish_reasons", [str(fr)])
            except Exception:
                pass

            usage = getattr(completion, "usage", None)
            if usage:
                span.set_attribute("gen_ai.usage.input_tokens", getattr(usage, "prompt_tokens", 0) or 0)
                span.set_attribute("gen_ai.usage.output_tokens", getattr(usage, "completion_tokens", 0) or 0)
                span.set_attribute("gen_ai.usage.total_tokens", getattr(usage, "total_tokens", 0) or 0)

            msg = completion.choices[0].message.content

            # ---- attach compact numbers to the chat span (for the table column) ----
            q = _last_user_message(messages)
            ctx = _extract_context_from_system(messages)
            try:
                basic = _basic_eval(answer_text=msg, question_text=q, context_text="")  # no embed calls here
                for k in ("ai.evaluation.answer_length", "ai.evaluation.sentences", "ai.evaluation.lexical_diversity"):
                    if k in basic:
                        span.set_attribute(k, float(basic[k]))
            except Exception:
                pass

            # ---- LLM-judge inside the chat span so we can emit events with this span active ----
            llm_metrics = {}
            if _HAS_LLM_EVAL:
                try:
                    llm_metrics = _llm_eval(question_text=q, answer_text=msg, context_text=ctx, ground_truth=None)

                    # Put canonical keys on span (helps the column show values)
                    if "ai.eval.coherence" in llm_metrics:
                        span.set_attribute("coherence", float(llm_metrics["ai.eval.coherence"]))
                        _emit_eval_event(span, "coherence", llm_metrics["ai.eval.coherence"],
                                         model or resp_mod, resp_id)
                    if "ai.eval.fluency" in llm_metrics:
                        span.set_attribute("fluency", float(llm_metrics["ai.eval.fluency"]))
                        _emit_eval_event(span, "fluency", llm_metrics["ai.eval.fluency"],
                                         model or resp_mod, resp_id)
                    # Optional QA metrics if you enable them later
                    if "ai.eval.qa.similarity" in llm_metrics:
                        span.set_attribute("similarity", float(llm_metrics["ai.eval.qa.similarity"]))
                        _emit_eval_event(span, "similarity", llm_metrics["ai.eval.qa.similarity"],
                                         model or resp_mod, resp_id)
                    if "ai.eval.qa.f1_score" in llm_metrics:
                        span.set_attribute("f1_score", float(llm_metrics["ai.eval.qa.f1_score"]))
                        _emit_eval_event(span, "f1_score", llm_metrics["ai.eval.qa.f1_score"],
                                         model or resp_mod, resp_id)
                except Exception:
                    pass

        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise

    # Sibling spans (keep, optional)
    try:
        with tracer.start_as_current_span("eval.metrics") as eval_span:
            eval_span.set_attribute("session.id", SESSION_ID)
            q = _last_user_message(messages)
            ctx = _extract_context_from_system(messages)
            metrics = _basic_eval(answer_text=msg, question_text=q, context_text=ctx)  # includes context_similarity
            for k, v in metrics.items():
                eval_span.set_attribute(k, float(v))
    except Exception:
        pass

    if _HAS_LLM_EVAL:
        try:
            with tracer.start_as_current_span("eval.llm") as llm_span:
                llm_span.set_attribute("session.id", SESSION_ID)
                q = _last_user_message(messages)
                ctx = _extract_context_from_system(messages)
                llm_metrics = _llm_eval(question_text=q, answer_text=msg, context_text=ctx, ground_truth=None)
                for k, v in llm_metrics.items():
                    if isinstance(v, (int, float)):
                        llm_span.set_attribute(k, float(v))
                    else:
                        llm_span.set_attribute(k, str(v))
        except Exception:
            pass

    return msg

