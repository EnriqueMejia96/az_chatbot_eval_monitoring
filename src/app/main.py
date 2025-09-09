import streamlit as st
import pandas as pd
import time
import os
from opentelemetry.trace import Status, StatusCode
from app.utils import get_context_from_query, get_response, text_embedding
from app.genaitools.telemetry.otel import init_telemetry
from app.genaitools.telemetry.events import emit_eval_event, set_genai_span_attrs
from app.genaitools.evals.runner import set_eval_env_vars, run_evaluators

from dotenv import load_dotenv
load_dotenv()

tracer, session_id, server_host = init_telemetry(
                                    project_endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT"),
                                    service_name = os.getenv("OTEL_SERVICE_NAME"),              
                                    capture_message_content = os.getenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true").lower() in ("1", "true", "yes"), 
                                    trace_to_console = os.getenv("OTEL_TRACE_TO_CONSOLE", "false").lower() in ("1", "true", "yes")
                                    )

set_eval_env_vars(
    judge_endpoint = os.getenv("AZURE_JUDGE_ENDPOINT"),
    judge_api_key= os.getenv("AZURE_JUDGE_API_KEY"),
    judge_deployment= os.getenv("AZURE_JUDGE_DEPLOYMENT_NAME"),
    judge_api_version= os.getenv("AZURE_JUDGE_API_VERSION"),
    subscription_id= os.getenv("AZURE_SUBSCRIPTION_ID"),
    resource_group= os.getenv("AZURE_RESOURCE_GROUP_NAME"),
    project_name= os.getenv("AZURE_PROJECT_NAME"),
    project_url= os.getenv("PROJECT_ENDPOINT")
)

df_vector_store = pd.read_pickle('df_vector_store.pkl')

def main_page():
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.0
    if "model" not in st.session_state:
        st.session_state.model = "gpt-4o"
    if "message_history" not in st.session_state:
        st.session_state.message_history = []

    with st.sidebar:
        st.header("CHAT Q&A :robot_face:")
        st.subheader("[LAB Evaluation & Monitoring]")
        st.subheader('Configuración del modelo :level_slider:')

        model_name = st.radio("**Elije un modelo**:", ("GPT-4o", "GPT-4o-mini", "GPT-o3-mini"))
        st.session_state.model = {
            "GPT-4o": "gpt-4o",
            "GPT-4o-mini": "gpt-4o-mini",
            "GPT-o3-mini": "gpt-o3-mini",
        }[model_name]

        st.session_state.temperature = st.slider(
            "**Nivel de creatividad de respuesta**  \n  [Poco creativo ►►► Muy creativo]",
            min_value=0.0, max_value=1.0, step=0.1, value=0.0
        )

    if st.session_state.get('generar_pressed', False):
        for m in st.session_state.message_history:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

    if prompt := st.chat_input("¿Cuál es tu consulta?"):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with tracer.start_as_current_span("testlab") as root:
                root.set_attribute("session.id", session_id)
                root.set_attribute("gen_ai.use_case", "chatbot_test")

                with tracer.start_as_current_span("operation.embedding") as span_embedding:
                    try:
                        t_embedding = time.time()
                        prompt_emb, embedding_dimension, input_tokens, total_tokens = text_embedding(input = prompt,
                                                                                                    model = os.environ["EMBEDDING_DEPLOYMENT"])
                        set_genai_span_attrs(
                            span = span_embedding,
                            server_host=server_host,
                            session_id=session_id,
                            service = 'openai',
                            operation = 'embedding',
                            model=os.environ["EMBEDDING_DEPLOYMENT"],                 
                            latency_ms = int((time.time()-t_embedding)*1000),
                            embedding_dimension = embedding_dimension,
                            input_tokens=input_tokens,
                            total_tokens=total_tokens
                        )
                    except Exception as e:
                        span_embedding.set_status(Status(StatusCode.ERROR, str(e)))
                        span_embedding.record_exception(e)
                        raise

                with tracer.start_as_current_span("operation.rag") as span_rag:
                    try:
                        t_rag = time.time()
                        rag_context , n_chunks= get_context_from_query(prompt_emb, df_vector_store, n_chunks=5)

                        set_genai_span_attrs(
                            span = span_rag,
                            server_host=server_host,
                            session_id=session_id,
                            service = 'custom_rag',
                            operation = 'rag',
                            latency_ms = int((time.time()-t_rag)*1000),
                            n_chunks = n_chunks,
                        )

                    except Exception as e:
                        span_rag.set_status(Status(StatusCode.ERROR, str(e)))
                        span_rag.record_exception(e)
                        raise

                with tracer.start_as_current_span("operation.chat.principal") as span_chat:
                    try:
                        t_chat = time.time()
                        full_response, input_tokens, output_tokens, total_tokens, resp_id = get_response(
                                                                                                model=st.session_state.model,
                                                                                                temperature=st.session_state.temperature,
                                                                                                rag_context=rag_context, 
                                                                                                prompt=prompt, 
                                                                                                history=st.session_state.message_history
                                                                                            )
                        set_genai_span_attrs(
                            span = span_chat,
                            server_host=server_host,
                            session_id=session_id,
                            service = 'openai',
                            operation = 'chat',
                            model=st.session_state.model,                 
                            temperature=st.session_state.temperature,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            total_tokens=total_tokens,
                            latency_ms = int((time.time()-t_chat)*1000)
                        )

                        evaluators = ["coherence", "fluency", "relevance", "indirect_attack"]
                        outs = run_evaluators(
                            evaluators = evaluators,
                            query=prompt,
                            response=full_response
                        )
                        print(str(outs))

                        for name, res in outs.items():
                            if "__error__" in res or "__skipped__" in res:
                                span_chat.add_event("gen_ai.evaluation.error", {"evaluator": name, "message": str(res)})
                                continue
                            emit_eval_event(
                                active_span=span_chat,
                                evaluator=name,
                                result=res,
                                model=st.session_state.model,
                                response_id=resp_id
                            )

                    except Exception as e:
                        span_chat.set_status(Status(StatusCode.ERROR, str(e)))
                        span_chat.record_exception(e)
                        raise

            message_placeholder.markdown(full_response)

        st.session_state.message_history += [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": full_response},
        ]
        st.session_state.generar_pressed = True

if __name__ == "__main__":
    main_page()
