import re

def last_user_message(messages):
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""

def extract_context_from_system(messages):
    if not messages:
        return ""
    sys = messages[0].get("content", "") if messages[0].get("role") == "system" else ""
    if not sys:
        return ""
    m = re.search(r"RESULTADOS DE BÚSQUEDA SEM[ÁA]NTICA:\s*(.+?)\n\s*Lee cuidadosamente", sys, flags=re.S)
    return m.group(1).strip() if m else ""

def basic_eval(answer_text: str, question_text: str, context_text: str):
    words = re.findall(r"\w+", answer_text, flags=re.UNICODE)
    length = len(answer_text)
    sentences = sum(answer_text.count(x) for x in [".","!","?"])
    lex_div = (len(set(w.lower() for w in words)) / max(1, len(words))) if words else 0.0
    return {
        "ai.evaluation.answer_length": float(length),
        "ai.evaluation.sentences": float(sentences),
        "ai.evaluation.lexical_diversity": float(lex_div),
    }
