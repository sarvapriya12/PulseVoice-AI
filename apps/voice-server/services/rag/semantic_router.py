import numpy as np
from typing import Literal, Tuple
from core.config import settings

RouteType = Literal["core_vitals", "chitchat", "booking_tool", "rag_knowledge"]

# Prototype utterances for each intent route
ROUTE_PROTOTYPES: dict[RouteType, list[str]] = {
    "core_vitals": [
        "what are your hours",
        "what time do you open tomorrow",
        "what time do you close",
        "are you open on saturday or sunday",
        "where are you located",
        "what is your clinic address",
        "how do i get there",
        "what is your phone number",
        "what insurance do you accept",
        "do you take aetna or bluecross",
        "do you accept medicare or medicaid",
        "what is your cancellation fee policy",
        "how much is the copay",
    ],
    "chitchat": [
        "hello",
        "hi there",
        "good morning",
        "good afternoon",
        "can you hear me",
        "thank you so much",
        "thanks for your help",
        "bye",
        "have a great day",
        "who are you",
        "what is your name",
    ],
    "booking_tool": [
        "i want to book an appointment",
        "schedule a visit with the doctor",
        "do you have any available slots on monday",
        "check appointment availability",
        "cancel my appointment",
        "i need to reschedule my booking",
        "can i get on the waitlist",
        "add me to the waitlist",
    ],
    "rag_knowledge": [
        "how should i prepare for my blood test",
        "can i eat before my fasting lab work",
        "what documents do i need to bring for intake",
        "what is your patient privacy notice",
        "how do you handle hipaa and patient confidentiality",
        "what should i bring to my first appointment",
        "what procedures do you perform at the clinic",
    ],
}

_ROUTER_INITIALIZED = False
_ROUTE_MATRICES: dict[RouteType, np.ndarray] = {}
_EMBEDDINGS_MODEL = None


def _get_router_embeddings():
    global _EMBEDDINGS_MODEL
    if _EMBEDDINGS_MODEL is None:
        from services.rag.embedder import _load_embeddings_singleton
        _EMBEDDINGS_MODEL = _load_embeddings_singleton()
    return _EMBEDDINGS_MODEL


def init_semantic_router():
    """Pre-compute and normalize route prototype vectors on startup."""
    global _ROUTER_INITIALIZED, _ROUTE_MATRICES
    if _ROUTER_INITIALIZED:
        return

    embeddings = _get_router_embeddings()
    for route, utterances in ROUTE_PROTOTYPES.items():
        vecs = embeddings.embed_documents(utterances)
        mat = np.array(vecs, dtype=np.float32)
        # Normalize each vector along axis 1
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        norm_mat = mat / norms
        _ROUTE_MATRICES[route] = norm_mat

    _ROUTER_INITIALIZED = True
    print(f"[SEMANTIC ROUTER] Initialized with {len(_ROUTE_MATRICES)} routes.")


def route_query(query: str, threshold: float = 0.50) -> Tuple[RouteType, float]:
    """
    Classify user query in <8ms using max-cosine similarity against route prototypes.
    Returns (route_name, confidence_score).
    """
    clean_q = query.strip().lower()
    if not clean_q:
        return "chitchat", 1.0

    if not _ROUTER_INITIALIZED:
        init_semantic_router()

    embeddings = _get_router_embeddings()
    q_vec = np.array(embeddings.embed_query(clean_q), dtype=np.float32)
    q_norm = np.linalg.norm(q_vec)
    if q_norm > 0:
        q_vec /= q_norm

    best_route: RouteType = "rag_knowledge"
    best_score = -1.0

    for route, mat in _ROUTE_MATRICES.items():
        # Dot product with each prototype utterance in this route
        scores = np.dot(mat, q_vec)
        route_max = float(np.max(scores))
        if route_max > best_score:
            best_score = route_max
            best_route = route

    # If score is below threshold, default to standard RAG knowledge fallback
    if best_score < threshold:
        return "rag_knowledge", best_score

    return best_route, best_score
