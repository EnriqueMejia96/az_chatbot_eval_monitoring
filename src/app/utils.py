import time, numpy as np, re
import pandas as pd  # if you need it here
from opentelemetry.trace import Status, StatusCode

from app.genaitools.config import settings
from app.genaitools.telemetry.otel import tracer, SESSION_ID, chat_client, server_host
from app.genaitools.telemetry.events import emit_eval_event
from app.genaitools.evals import basic as ev_basic
from app.genaitools.evals import llm_judge as ev_judge
from app.genaitools.evals import safety as ev_safety

# ---- Embeddings ----
def text_embedding(text=[]):
    with tracer.start_as_current_span("embeddings.create") as span:
        span.set_attribute("session.id", SESSION_ID)
        span.set_attribute("gen_ai.system", "openai")
        span.set_attribute("gen_ai.operation.name", "embedding")
        span.set_attribute("gen_ai.request.model", settings.embedding_deployment)
        span.set_attribute("server.address", server_host)
        try:
            t0 = time.time()
            resp = chat_client.embeddings.create(
                model=settings.embedding_deployment,
                input=text,
                encoding_format="float",
            )
            span.set_attribute("gen_ai.response.latency_ms", int((time.time()-t0)*1000))
            emb = resp.data[0].embedding
            span.set_attribute("embedding.dimension", len(emb))
            if hasattr(resp, "id"):    span.set_attribute("gen_ai.response.id", resp.id)
            if hasattr(resp, "model"): span.set_attribute("gen_ai.response.model", resp.model)
            usage = getattr(resp, "usage", None)
            if usage:
                span.set_attribute("gen_ai.usage.input_tokens",  getattr(usage, "prompt_tokens", 0) or 0)
                span.set_attribute("gen_ai.usage.total_tokens",  getattr(usage, "total_tokens", 0) or 0)
            return emb
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise

# ---- Retrieval (no globals) ----
def _cosine(row_vec, q_vec):
    denom = (np.linalg.norm(row_vec) * np.linalg.norm(q_vec)) or 1.0
    return float(np.dot(row_vec, q_vec) / denom)

def get_context_from_query(query, vector_store, n_chunks=5):
    with tracer.start_as_current_span("rag.select_context") as span:
        span.set_attribute("session.id", SESSION_ID)
        span.set_attribute("rag.top_k", n_chunks)

        q_vec = np.array(text_embedding(query), dtype=np.float32)
        sims = vector_store["Embedding"].apply(lambda row: _cosine(row, q_vec))
        top_idx = sims.sort_values(ascending=False).head(n_chunks).index
        chunks = list(vector_store.loc[top_idx, "Chunks"])
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

def get_response(model, temperature, messages):
    with tracer.start_as_current_span("chat.completions.create") as span:
        span.set_attribute("session.id", SESSION_ID)
        span.set_attribute("gen_ai.system", "openai")
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.request.temperature", float(temperature))
        span.set_attribute("server.address", server_host)
        try:
            t0 = time.time()
            completion = chat_client.chat.completions.create(model=model, temperature=temperature, messages=messages)
            span.set_attribute("gen_ai.response.latency_ms", int((time.time()-t0)*1000))
            msg = completion.choices[0].message.content
            resp_id  = getattr(completion, "id", "") or ""
            resp_mod = getattr(completion, "model", "") or ""

            # basic metrics
            q   = ev_basic.last_user_message(messages)
            ctx = ev_basic.extract_context_from_system(messages)
            basic = ev_basic.basic_eval(answer_text=msg, question_text=q, context_text="")
            for k in ("ai.evaluation.answer_length","ai.evaluation.sentences","ai.evaluation.lexical_diversity"):
                if k in basic:
                    span.set_attribute(k, float(basic[k]))

            # LLM-judge
            from app.genaitools.config import settings as _s  # local alias
            if getattr(ev_judge, "_HAS_LLM_EVAL", False) and _s.eval_attach_to_chat:
                try:
                    llm_metrics = ev_judge.llm_eval(q, msg, ctx, ground_truth=None)
                    if "ai.eval.coherence" in llm_metrics:
                        span.set_attribute("coherence", float(llm_metrics["ai.eval.coherence"]))
                        emit_eval_event(span, "coherence", llm_metrics["ai.eval.coherence"], model or resp_mod, resp_id)
                    if "ai.eval.fluency" in llm_metrics:
                        span.set_attribute("fluency", float(llm_metrics["ai.eval.fluency"]))
                        emit_eval_event(span, "fluency", llm_metrics["ai.eval.fluency"], model or resp_mod, resp_id)
                    if "ai.eval.qa.similarity" in llm_metrics:
                        span.set_attribute("similarity", float(llm_metrics["ai.eval.qa.similarity"]))
                        emit_eval_event(span, "similarity", llm_metrics["ai.eval.qa.similarity"], model or resp_mod, resp_id)
                    if "ai.eval.qa.f1_score" in llm_metrics:
                        span.set_attribute("f1_score", float(llm_metrics["ai.eval.qa.f1_score"]))
                        emit_eval_event(span, "f1_score", llm_metrics["ai.eval.qa.f1_score"], model or resp_mod, resp_id)
                except Exception:
                    pass

            # Safety
            if ev_safety.HAS_SAFETY and _s.eval_safety_enable:
                try:
                    safety = ev_safety.safety_eval(q, msg, ctx)
                    for key, score in safety.items():
                        name = key.rsplit(".", 1)[-1]
                        span.set_attribute(f"safety.{name}", float(score))
                        # if you want: emit_eval_event(span, name, float(score), model or resp_mod, resp_id)
                except Exception:
                    pass

        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise

    # separate span with context-sim (optional to avoid extra embed calls in the main span)
    try:
        with tracer.start_as_current_span("eval.metrics") as eval_span:
            eval_span.set_attribute("session.id", SESSION_ID)
            q   = ev_basic.last_user_message(messages)
            ctx = ev_basic.extract_context_from_system(messages)

            # If you still want context similarity via embeddings, reuse text_embedding here
            # Only compute if settings.eval_context_sim is True and ctx exists
            # (left as-is to keep your original behavior)
            # ...
    except Exception:
        pass

    return msg
