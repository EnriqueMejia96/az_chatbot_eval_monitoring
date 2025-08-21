import os
import uuid
import numpy as np
from dotenv import load_dotenv

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
    raise RuntimeError(
        "PROJECT_ENDPOINT must be set as environment variables."
    )

# ---------- OpenTelemetry ----------
tracer = trace.get_tracer(__name__)
SESSION_ID = str(uuid.uuid4())

# Capture prompts/outputs in telemetry (careful for PII; you have PII checks elsewhere)
os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true"

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
try:
    ai_conn = project_client.telemetry.get_application_insights_connection_string()
    if ai_conn:
        configure_azure_monitor(connection_string=ai_conn)
except Exception:
    # Non-fatal: continue without centralized telemetry if not configured
    pass

# Instrument OpenAI-compatible client
OpenAIInstrumentor().instrument()

# OpenAI-compatible client from the Project
chat_client = project_client.get_openai_client(api_version="2024-10-21")

# ---------- Your original helpers (kept intact API) ----------
def text_embedding(text=[]):
    with tracer.start_as_current_span("embedding.create") as span:
        span.set_attribute("session.id", SESSION_ID)
        try:
            resp = chat_client.embeddings.create(
                model=embedding_deployment,
                input=text,
                encoding_format="float",
            )
            return resp.data[0].embedding
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
    global query_vector
    query_vector = np.array(text_embedding(query))
    top_matched = (
        vector_store["Embedding"]
        .apply(cosine_similarity)
        .sort_values(ascending=False)[:n_chunks]
        .index
    )
    top_matched_df = vector_store[vector_store.index.isin(top_matched)][["Chunks"]]
    return list(top_matched_df["Chunks"])

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
        try:
            completion = chat_client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=messages,
            )
            return completion.choices[0].message.content
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
