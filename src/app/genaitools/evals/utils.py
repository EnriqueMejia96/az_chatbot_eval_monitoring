import inspect

def safe_call_eval(ev, **kwargs):
    try:
        sig = inspect.signature(ev.__call__)
        return ev(**{k:v for k,v in kwargs.items() if k in sig.parameters})
    except Exception:
        try:
            sig = inspect.signature(ev)
            return ev(**{k:v for k,v in kwargs.items() if k in sig.parameters})
        except Exception:
            try:
                return ev(query=kwargs.get("query"), response=kwargs.get("response"),
                          context=kwargs.get("context"), ground_truth=kwargs.get("ground_truth"))
            except Exception:
                return {}

def extract_numeric(out: dict, primary_key: str) -> float | None:
    if not isinstance(out, dict):
        return None
    for k in (primary_key, f"{primary_key}_score", "value"):
        if k in out and isinstance(out[k], (int,float)):
            return float(out[k])
    return None
