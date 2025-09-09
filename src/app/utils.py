from app.genaitools.telemetry.otel import chat_client
import numpy as np

# ---- Embeddings ----
def text_embedding(input, model):
    resp = chat_client.embeddings.create(
        model=model,
        input=input,
        encoding_format="float",
    )
    
    emb = resp.data[0].embedding
    embedding_dimension = len(emb)

    usage = getattr(resp, "usage", None)
    if usage:
        input_tokens = getattr(usage, "prompt_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", 0)
    return emb, embedding_dimension, input_tokens, total_tokens

# ---- Retrieval ----
def _cosine(row_vec, q_vec):
    denom = (np.linalg.norm(row_vec) * np.linalg.norm(q_vec)) or 1.0
    return float(np.dot(row_vec, q_vec) / denom)

def get_context_from_query(prompt_emb, vector_store, n_chunks=5):
    q_vec = np.array(prompt_emb, dtype=np.float32)
    sims = vector_store["Embedding"].apply(lambda row: _cosine(row, q_vec))
    top_idx = sims.sort_values(ascending=False).head(n_chunks).index
    chunks = list(vector_store.loc[top_idx, "Chunks"])

    return chunks, n_chunks

# ---- Chat response ----

custom_prompt = """
Eres una Inteligencia Artificial super avanzada que trabaja asistente personal.
Utilice los RESULTADOS DE BÚSQUEDA SEMANTICA para responder las preguntas del usuario. 
Solo debes utilizar la informacion de la BUSQUEDA SEMANTICA si es que hace sentido y tiene relacion con la pregunta del usuario.
Si la respuesta no se encuentra dentro del contexto de la búsqueda semántica, no inventes una respuesta, y responde amablemente que no tienes información para responder.

RESULTADOS DE BÚSQUEDA SEMANTICA:
{source}

Lee cuidadosamente las instrucciones, respira profundo y escribe una respuesta para el usuario!
"""

def extract_usage(resp):
    u = getattr(resp, "usage", None)
    if u is None:
        return None, None, None
    def get(name):
        return getattr(u, name, None) if hasattr(u, name) else (u.get(name) if isinstance(u, dict) else None)

    input_tokens  = get("prompt_tokens")    or get("input_tokens")
    output_tokens = get("completion_tokens") or get("output_tokens")
    total_tokens  = get("total_tokens") or ((input_tokens or 0) + (output_tokens or 0))
    return input_tokens, output_tokens, total_tokens


def get_response(model, temperature, rag_context, prompt, history):

    messages = (
        [{"role": "system", "content": f"{custom_prompt.format(source=str(rag_context))}"}]
        + history
        + [{"role": "user", "content": prompt}]
    )
    completion = chat_client.chat.completions.create(model=model, temperature=temperature, messages=messages)
    input_tokens, output_tokens, total_tokens = extract_usage(completion)

    msg = completion.choices[0].message.content
    resp_id  = getattr(completion, "id", "") or ""

    return msg, input_tokens, output_tokens, total_tokens, resp_id
