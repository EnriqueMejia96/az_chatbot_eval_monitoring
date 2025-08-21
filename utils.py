import os
import uuid
import time
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

if not project_endpoint:
    raise RuntimeError("PROJECT_ENDPOINT must be set as environment variables.")

# Helpful service name + content capture (consider disabling in prod)
os.environ.setdefault("OTEL_SERVICE_NAME", "rag-app")
os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true")

# ---------- OpenTelemetry ----------
tracer = trace.get_tracer(__name__)
SESSION_ID = str(uuid.uuid4())

# For trace attributes like server.address
_server_host = urlparse(project_endpoint).hostname or ""

# ---------- Azure AI Project & Observability ----------
# Allow both Env creds (e.g., local dev with AZURE_* vars) and Managed Identity in App Service.
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

# Wire up Azure Monitor automatically (Application Insights is behind the project)
ai_conn = project_client.telemetry.get_application_insights_connection_string()
if not ai_conn:
    raise RuntimeError("No App Insights connected to this Project – configure it in the portal (Tracing).")
print("AppInsights conn str present:", bool(ai_conn))
configure_azure_monitor(connection_string=ai_conn)


# Optional: also export traces to console (useful for CI). Enable with OTEL_TRACE_TO_CONSOLE=true
if os.getenv("OTEL_TRACE_TO_CONSOLE", "false").lower() in ("1", "true", "yes"):
    try:
        from opentelemetry import trace as trace_api
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
        provider = trace_api.get_tracer_provider()
        if hasattr(provider, "add_span_processor"):
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    except Exception:
        # Don't fail app if console exporter can't be attached
        pass

# Instrument OpenAI-compatible client
OpenAIInstrumentor().instrument()

# OpenAI-compatible client from the Project
chat_client = project_client.get_openai_client(api_version="2024-10-21")

# ---------- Your original helpers (kept intact API) ----------
def text_embedding(text=[]):
    # Trace-only: richer span name & attrs; no logic change
    with tracer.start_as_current_span("embeddings.create") as span:
        span.set_attribute("session.id", SESSION_ID)
        span.set_attribute("gen_ai.system", "openai")  # per doc examples
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

            # Response metadata if available
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

def get_response(model, temperature, messages):
    # Trace-only: add standard GenAI attrs and token/response metadata; no logic change
    with tracer.start_as_current_span("chat.completions.create") as span:
        span.set_attribute("session.id", SESSION_ID)
        span.set_attribute("gen_ai.system", "openai")  # per doc examples
        span.set_attribute("gen_ai.operation.name", "chat")  # align with docs
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.request.temperature", float(temperature))
        span.set_attribute("server.address", _server_host)
        try:
            t0 = time.time()
            completion = chat_client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=messages,
            )
            dt_ms = int((time.time() - t0) * 1000)
            span.set_attribute("gen_ai.response.latency_ms", dt_ms)

            # Response metadata (if exposed by SDK)
            if hasattr(completion, "id"):
                span.set_attribute("gen_ai.response.id", completion.id)
            if hasattr(completion, "model"):
                span.set_attribute("gen_ai.response.model", completion.model)

            # Finish reason (first choice)
            try:
                fr = completion.choices[0].finish_reason
                if fr is not None:
                    span.set_attribute("gen_ai.response.finish_reasons", [str(fr)])
            except Exception:
                pass

            # Capture usage if provided by SDK
            usage = getattr(completion, "usage", None)
            if usage:
                span.set_attribute("gen_ai.usage.input_tokens", getattr(usage, "prompt_tokens", 0) or 0)
                span.set_attribute("gen_ai.usage.output_tokens", getattr(usage, "completion_tokens", 0) or 0)
                span.set_attribute("gen_ai.usage.total_tokens", getattr(usage, "total_tokens", 0) or 0)

            return completion.choices[0].message.content
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
