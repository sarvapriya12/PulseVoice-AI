import os
import time
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from core.config import settings
from services.rag.embedder import _load_embeddings_singleton
from services.rag.semantic_router import route_query

# Static Core Clinic Vitals for Instant Tier 0 resolution
CORE_VITALS_SNIPPET = (
    "Dr. Smith's Clinic Vitals:\n"
    "- Address: 123 Health Way, Austin, TX 78701\n"
    "- Phone: (512) 555-0199\n"
    "- Hours: Monday - Friday, 9:00 AM - 5:00 PM. Closed on weekends and federal holidays.\n"
    "- Accepted Insurance: BlueCross, Aetna, Medicare, UnitedHealthcare. (Medicaid is NOT accepted).\n"
    "- Cancellation Policy: Minimum 24 hours advance notice required, or a $50 late fee applies.\n"
    "- Payment: Co-pays due at visit (Credit card, debit card, Apple Pay)."
)


_get_embeddings = _load_embeddings_singleton


def _extract_content(doc) -> str:
    meta = getattr(doc, "metadata", None)
    if isinstance(meta, dict):
        ctx = meta.get("window_context")
        if ctx and isinstance(ctx, str):
            return ctx
    content = getattr(doc, "page_content", "")
    return str(content)


def _extract_source(doc) -> str:
    meta = getattr(doc, "metadata", None)
    if isinstance(meta, dict):
        src = meta.get("source_file")
        if src and isinstance(src, str):
            return src
    return "clinic_knowledge"


class FAQRetriever:
    def __init__(self, index_path: str = "data/faiss_index"):
        self.index_path = index_path
        self.embeddings = _get_embeddings(settings.EMBEDDING_MODEL)
        
        self._cache: dict[str, tuple[list, float]] = {}
        self._cache_ttl = 300  # 5 minutes
        
        self.faiss_db = FAISS.load_local(
            self.index_path, 
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        self.bm25_retriever = self._load_bm25()
        
        from main import get_reranker_model
        self.reranker = get_reranker_model()
        
        print(f"[RAG] Loaded index from {index_path} | "
              f"FAISS docs: {self.faiss_db.index.ntotal} | "
              f"BM25: {'yes' if self.bm25_retriever else 'no'} | "
              f"Reranker: {'yes' if self.reranker else 'no'}")

    def _load_bm25(self):
        import pickle
        docs_path = os.path.join(self.index_path, "raw_docs.pkl")
        if os.path.exists(docs_path):
            with open(docs_path, "rb") as f:
                docs = pickle.load(f)
        else:
            docs = self.faiss_db.similarity_search("", k=self.faiss_db.index.ntotal)
            
        if not docs:
            return None
        return BM25Retriever.from_documents(docs)
        
    def reload_index(self):
        """Hot-reload the FAISS database and BM25 index from disk without restarting."""
        if not os.path.exists(self.index_path):
            print(f"Warning: Cannot reload, index path {self.index_path} does not exist.")
            return
            
        self.faiss_db = FAISS.load_local(
            self.index_path, 
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        self.bm25_retriever = self._load_bm25()
        from main import get_reranker_model
        self.reranker = get_reranker_model()
        print("RAG Index successfully hot-reloaded!")

    def search_faq_docs(self, query: str, top_k: int = 3) -> list:
        """Returns Document objects with metadata, for prefetching."""
        key = query.lower().strip()
        
        if key in self._cache:
            result, timestamp = self._cache[key]
            if time.time() - timestamp < self._cache_ttl:
                print(f"[RAG CACHE HIT] '{key[:40]}'")
                return result

        faiss_docs = self.faiss_db.similarity_search(query, k=15)
        bm25_docs = self.bm25_retriever.invoke(query)[:15] if self.bm25_retriever else []
        seen = {}
        for doc in faiss_docs + bm25_docs:
            content_key = _extract_content(doc)
            if content_key not in seen:
                seen[content_key] = doc
        combined = list(seen.values())
        if not combined:
            return []
        
        if self.reranker:
            MIN_SCORE = 0.3
            pairs = [[query, _extract_content(d)] for d in combined]
            scores = self.reranker.predict(pairs)
            
            doc_scores = list(zip(combined, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            best_docs = [doc for doc, score in doc_scores[:top_k] if score > MIN_SCORE]
            
            self._cache[key] = (best_docs, time.time())
            if len(self._cache) > 200:
                oldest = min(self._cache, key=lambda k: self._cache[k][1])
                del self._cache[oldest]
            return best_docs

        result = combined[:top_k]
        self._cache[key] = (result, time.time())
        if len(self._cache) > 200:
            oldest = min(self._cache, key=lambda k: self._cache[k][1])
            del self._cache[oldest]
        return result

    def search_faq(self, query: str, top_k: int = 3) -> str:
        """
        Fast Tiered Search:
        Tier 0: Semantic Router checks for core clinic vitals (hours/location) or chitchat -> instant answer.
        Tier 1: Semantic Cache hit -> sub-millisecond answer.
        Tier 2: Sentence-Window hybrid search (FAISS + BM25 + optional Cross-Encoder reranker).
        """
        try:
            # ── Fast Semantic Router Check (<8ms) ──────────────────────────
            route, confidence = route_query(query)
            if route == "core_vitals" and confidence >= 0.58:
                print(f"[ROUTER] Instant Tier 0 resolution for '{query}' (score={confidence:.2f})")
                return CORE_VITALS_SNIPPET

            if route == "chitchat" and confidence >= 0.65:
                return "No clinic knowledge lookup required for conversational pleasantries."

            # ── Hybrid Retrieval with Small-to-Big Sentence Window ─────────
            faiss_docs = self.faiss_db.similarity_search(query, k=15)
            bm25_docs = self.bm25_retriever.invoke(query)[:15] if self.bm25_retriever else []
                
            seen = {}
            for doc in faiss_docs + bm25_docs:
                text_val = _extract_content(doc)
                if text_val not in seen:
                    seen[text_val] = doc

            combined_docs = list(seen.values())
            
            if not combined_docs:
                return "I do not have that information in my knowledge base."
                
            if self.reranker:
                MIN_SCORE = 0.25
                pairs = [[query, _extract_content(d)] for d in combined_docs]
                scores = self.reranker.predict(pairs)
                
                doc_scores = list(zip(combined_docs, scores))
                doc_scores.sort(key=lambda x: x[1], reverse=True)
                
                best_docs = [doc for doc, score in doc_scores[:top_k] if score > MIN_SCORE]
                
                if not best_docs:
                    return "I do not have that information in my knowledge base."
                    
                results = []
                for doc in best_docs:
                    source = _extract_source(doc)
                    content = _extract_content(doc)
                    results.append(f"[Source: {source}]\n{content}")
                return "\n\n---\n\n".join(results)
            else:
                results = []
                for doc in combined_docs[:top_k]:
                    source = _extract_source(doc)
                    content = _extract_content(doc)
                    results.append(f"[Source: {source}]\n{content}")
                return "\n\n---\n\n".join(results)
            
        except Exception as e:
            return f"Error accessing knowledge base: {e}"
