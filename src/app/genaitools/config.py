import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    project_endpoint: str = os.environ["PROJECT_ENDPOINT"]
    embedding_deployment: str = os.environ["EMBEDDING_DEPLOYMENT"]
    eval_attach_to_chat: bool = os.getenv("EVAL_ATTACH_TO_CHAT", "true").lower() in ("1","true","yes")
    eval_context_sim:    bool = os.getenv("EVAL_CONTEXT_SIMILARITY","true").lower() in ("1","true","yes")
    eval_threshold:      int  = int(os.getenv("EVAL_THRESHOLD","3"))
    eval_qa_enable:      bool = os.getenv("EVAL_QA_ENABLE","false").lower() in ("1","true","yes")
    eval_safety_enable:  bool = os.getenv("EVAL_SAFETY_ENABLE","true").lower() in ("1","true","yes")

settings = Settings()
