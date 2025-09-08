import os, uuid
from urllib.parse import urlparse
from opentelemetry import trace
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor
from . import __init__  # keep package import happy
from app.genaitools.config import settings

# friendly service name + content capture
os.environ.setdefault("OTEL_SERVICE_NAME", "rag-app")
os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true")

tracer = trace.get_tracer(__name__)
SESSION_ID = str(uuid.uuid4())
_server_host = urlparse(settings.project_endpoint).hostname or ""

# Azure creds + project client
credential = DefaultAzureCredential(
    exclude_environment_credential=False,
    exclude_managed_identity_credential=False,
    exclude_shared_token_cache_credential=True,
    exclude_visual_studio_code_credential=True,
    exclude_powershell_credential=True,
    exclude_cli_credential=False,
)
project_client = AIProjectClient(credential=credential, endpoint=settings.project_endpoint)

# App Insights hookup
ai_conn = project_client.telemetry.get_application_insights_connection_string()
if not ai_conn:
    raise RuntimeError("No App Insights connected – configure it in the Project (Tracing).")
configure_azure_monitor(connection_string=ai_conn)

# Optional console exporter
if os.getenv("OTEL_TRACE_TO_CONSOLE", "false").lower() in ("1","true","yes"):
    try:
        from opentelemetry import trace as trace_api
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
        provider = trace_api.get_tracer_provider()
        if hasattr(provider, "add_span_processor"):
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    except Exception:
        pass

# Instrument OpenAI-compatible client once
OpenAIInstrumentor().instrument()
chat_client = project_client.get_openai_client(api_version="2024-10-21")

# convenience exports
server_host = _server_host
