from __future__ import annotations

import os
import uuid
from typing import Optional, Mapping, Any, Tuple
from urllib.parse import urlparse

from opentelemetry import trace
from opentelemetry.trace import Tracer
from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor

from azure.identity import DefaultAzureCredential
from azure.core.credentials import TokenCredential
from azure.ai.projects import AIProjectClient
from azure.monitor.opentelemetry import configure_azure_monitor

_TELEMETRY_READY = False
_TRACER: Optional[Tracer] = None
_SESSION_ID: Optional[str] = None
_SERVER_HOST: str = ""


def init_telemetry(
    *,
    appinsights_connection_string: Optional[str] = None,
    project_endpoint: Optional[str] = None,
    credential: Optional[TokenCredential] = None,
    default_credential_kwargs: Optional[Mapping[str, Any]] = None,
    service_name: Optional[str] = None,
    capture_message_content: Optional[bool] = None,
    trace_to_console: Optional[bool] = None,
) -> Tuple[Tracer, str, str]:
    """
    One-time, idempotent OpenTelemetry setup for the process.
    Returns (tracer, SESSION_ID, server_host).
    Safe to call multiple times; subsequent calls return cached values.
    """
    global _TELEMETRY_READY, _TRACER, _SESSION_ID, _SERVER_HOST
    if _TELEMETRY_READY and _TRACER and _SESSION_ID is not None:
        return _TRACER, _SESSION_ID, _SERVER_HOST

    resolved_service_name = service_name or os.getenv("OTEL_SERVICE_NAME") or "chatbot-app"
    os.environ["OTEL_SERVICE_NAME"] = resolved_service_name

    resolved_capture = (
        capture_message_content
        if capture_message_content is not None
        else os.getenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true").lower() in ("1", "true", "yes")
    )
    os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true" if resolved_capture else "false"

    resolved_trace_to_console = (
        trace_to_console
        if trace_to_console is not None
        else os.getenv("OTEL_TRACE_TO_CONSOLE", "false").lower() in ("1", "true", "yes")
    )

    resolved_endpoint = project_endpoint or os.getenv("AZURE_AI_PROJECT_ENDPOINT") or ""

    conn_str = appinsights_connection_string or os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")

    if not conn_str:
        if not resolved_endpoint:
            raise RuntimeError(
                "No Application Insights connection string available. "
                "Pass `appinsights_connection_string`, set APPLICATIONINSIGHTS_CONNECTION_STRING, "
                "or provide `project_endpoint`/AZURE_AI_PROJECT_ENDPOINT so it can be fetched from the Project."
            )

        if credential is None:
            dck = dict(
                exclude_environment_credential=False,
                exclude_managed_identity_credential=False,
                exclude_shared_token_cache_credential=True,
                exclude_visual_studio_code_credential=True,
                exclude_powershell_credential=True,
                exclude_cli_credential=False,
            )
            if default_credential_kwargs:
                dck.update(default_credential_kwargs)
            credential = DefaultAzureCredential(**dck)

        proj = AIProjectClient(credential=credential, endpoint=resolved_endpoint)
        conn_str = proj.telemetry.get_application_insights_connection_string()

    if not conn_str:
        raise RuntimeError("Application Insights connection string could not be resolved from any source.")

    configure_azure_monitor(connection_string=conn_str)

    try:
        OpenAIInstrumentor().instrument()
    except Exception:
        pass

    if resolved_trace_to_console:
        try:
            from opentelemetry import trace as trace_api
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
            provider = trace_api.get_tracer_provider()
            if hasattr(provider, "add_span_processor"):
                provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        except Exception:
            pass

    _TRACER = trace.get_tracer(__name__)
    _SESSION_ID = _SESSION_ID or str(uuid.uuid4())
    _SERVER_HOST = urlparse(resolved_endpoint).hostname or ""

    _TELEMETRY_READY = True
    return _TRACER, _SESSION_ID, _SERVER_HOST


def make_openai_client(
    *,
    project_endpoint: Optional[str] = None,
    api_version: str = "2024-10-21",
    credential: Optional[TokenCredential] = None,
    default_credential_kwargs: Optional[Mapping[str, Any]] = None,
) -> Any:
    """
    Factory: returns an OpenAI-compatible client bound to the given Azure AI Project.
    Does NOT touch global telemetry; call `init_telemetry()` separately if you need it.
    """
    endpoint = project_endpoint or os.getenv("AZURE_AI_PROJECT_ENDPOINT")
    if not endpoint:
        raise ValueError("Project endpoint is required. Pass `project_endpoint` or set AZURE_AI_PROJECT_ENDPOINT.")

    if credential is None:
        dck = dict(
            exclude_environment_credential=False,
            exclude_managed_identity_credential=False,
            exclude_shared_token_cache_credential=True,
            exclude_visual_studio_code_credential=True,
            exclude_powershell_credential=True,
            exclude_cli_credential=False,
        )
        if default_credential_kwargs:
            dck.update(default_credential_kwargs)
        credential = DefaultAzureCredential(**dck)

    project_client = AIProjectClient(credential=credential, endpoint=endpoint)
    return project_client.get_openai_client(api_version=api_version)


def get_instrumented_openai_client(**kwargs) -> Any:
    """
    Convenience: init telemetry (no-op if already done) and return an OpenAI-compatible client.
    Accepts the union of kwargs from both functions.
    """
    init_kwargs = {k: kwargs.get(k) for k in (
        "appinsights_connection_string", "project_endpoint", "credential",
        "default_credential_kwargs", "service_name",
        "capture_message_content", "trace_to_console"
    )}
    tracer, session_id, server_host = init_telemetry(**init_kwargs)

    make_kwargs = {k: kwargs.get(k) for k in (
        "project_endpoint", "api_version", "credential",
        "default_credential_kwargs"
    )}
    client = make_openai_client(**make_kwargs)
    return client
