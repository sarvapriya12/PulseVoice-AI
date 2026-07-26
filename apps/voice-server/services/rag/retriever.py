import os
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from core.config import settings

def _get_embeddings():
    return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

class FAQRetriever:
    def __init__(self, index_path: str = "data/faiss_index"):
        """
        Initialize the FAQRetriever.
        
        Args:
            index_path (str): The path to the FAISS index.
            
        Note:
            allow_dangerous_deserialization=True is required for loading local FAISS indexes.
            For BM25, we grab the docs directly from the FAISS docstore for testing purposes.
        """
        self.index_path = index_path
        self.embeddings = _get_embeddings()
        
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
        import os
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
        import time
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
            if doc.page_content not in seen:
                seen[doc.page_content] = doc
        combined = list(seen.values())
        if not combined:
            return []
        
        if self.reranker:
            MIN_SCORE = 0.3
            pairs = [[query, d.page_content] for d in combined]
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
        Search the FAQ knowledge base for a given query.
        
        This method uses a two-stage retrieval process:
        Stage 1: The Broad Net - Grabs top 15 results from FAISS and BM25, combines them, and removes duplicates.
        Stage 2: The Sniper (Cross-Encoder) - Uses a cross-encoder model to score query-document pairs 
                 and returns the absolute best `top_k` documents. Falls back to combined results if the reranker fails.
                 
        Args:
            query (str): The search query.
            top_k (int): The number of top documents to return. Default is 3.
            
        Returns:
            str: A concatenated string of the best documents, or an error message.
        """
        try:
            faiss_docs = self.faiss_db.similarity_search(query, k=15)
            
            bm25_docs = []
            if self.bm25_retriever:
                bm25_docs = self.bm25_retriever.invoke(query)[:15]
                
            seen = {}
            for doc in faiss_docs + bm25_docs:
                if doc.page_content not in seen:
                    seen[doc.page_content] = doc

            combined_docs = list(seen.values())
            
            if not combined_docs:
                return "I do not have that information in my knowledge base."
                
            if self.reranker:
                MIN_SCORE = 0.3
                pairs = [[query, doc.page_content] for doc in combined_docs]
                scores = self.reranker.predict(pairs)
                
                doc_scores = list(zip(combined_docs, scores))
                doc_scores.sort(key=lambda x: x[1], reverse=True)
                
                best_docs = [doc for doc, score in doc_scores[:top_k] if score > MIN_SCORE]
                
                if not best_docs:
                    return "I do not have that information in my knowledge base."
                    
                results = []
                for doc in best_docs:
                    source = doc.metadata.get("source_file", "unknown")
                    results.append(f"[Source: {source}]\n{doc.page_content}")
                return "\n\n---\n\n".join(results)
            else:
                results = []
                for doc in combined_docs[:top_k]:
                    source = doc.metadata.get("source_file", "unknown")
                    results.append(f"[Source: {source}]\n{doc.page_content}")
                return "\n\n---\n\n".join(results)
            
        except Exception as e:
            return f"Error accessing knowledge base: {e}"
