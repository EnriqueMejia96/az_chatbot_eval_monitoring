import streamlit as st
from openai import OpenAI
import pandas as pd
from utils import get_context_from_query, custom_prompt, get_response, tracer, SESSION_ID

df_vector_store = pd.read_pickle('df_vector_store.pkl')

def main_page():
  if "temperature" not in st.session_state:
      st.session_state.temperature = 0.0
  if "model" not in st.session_state:
      st.session_state.model = "gpt-4o"
  if "message_history" not in st.session_state:
      st.session_state.message_history = []

  with st.sidebar:
    st.header(body="CHAT Q&A :robot_face:")
    st.subheader(body="[LAB Evaluation & Monitoring]")
    st.header(body="")
    st.subheader('Configuración del modelo :level_slider:')

    model_name = st.radio("**Elije un modelo**:", ("GPT-4o", "GPT-4o-mini", "GPT-o3-mini"))
    if model_name == "GPT-4o":
      st.session_state.model = "gpt-4o"
    elif model_name == "GPT-4o-mini":
      st.session_state.model = "gpt-4o-mini"
    elif model_name == "GPT-o3-mini":
      st.session_state.model = "gpt-o3-mini"
    
    st.session_state.temperature = st.slider("**Nivel de creatividad de respuesta**  \n  [Poco creativo ►►► Muy creativo]",
                                             min_value = 0.0,
                                             max_value = 1.0,
                                             step      = 0.1,
                                             value     = 0.0)
    
  if st.session_state.get('generar_pressed', False):
    for message in st.session_state.message_history:
      with st.chat_message(message["role"]):
        st.markdown(message["content"])

  if prompt := st.chat_input("¿Cuál es tu consulta?"):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        with tracer.start_as_current_span("rag.qa") as root:
            root.set_attribute("session.id", SESSION_ID)
            root.set_attribute("gen_ai.use_case", "rag_qa")

            Context_List = get_context_from_query(
                query=prompt,
                vector_store=df_vector_store,
                n_chunks=5
            )
            messages = [{"role": "system", "content": f"{custom_prompt.format(source=str(Context_List))}"}] \
                      + st.session_state.message_history \
                      + [{"role": "user", "content": prompt}]

            full_response = get_response(
                model=st.session_state.model,
                temperature=st.session_state.temperature,
                messages=messages
            )

        message_placeholder.markdown(full_response)

    st.session_state.message_history.append({"role": "user", "content": prompt})
    st.session_state.message_history.append({"role": "assistant", "content": full_response})
    st.session_state.generar_pressed = True

if __name__ == "__main__":
    main_page()
