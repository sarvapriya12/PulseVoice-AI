import os
import re
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

_GLOBAL_EMBEDDINGS = None

class CachedHuggingFaceEmbeddings(Embeddings):
    """Embedding cache layer to eliminate redundant vectorization compute."""
    def __init__(self, base_embeddings, max_cache_size: int = 1000):
        self.base = base_embeddings
        self._cache: dict[str, list[float]] = {}
        self._max_cache_size = max_cache_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results = []
        missing = []
        missing_indices = []
        for idx, text in enumerate(texts):
            clean = text.strip()
            if clean in self._cache:
                results.append(self._cache[clean])
            else:
                results.append(None)
                missing.append(clean)
                missing_indices.append(idx)

        if missing:
            computed = self.base.embed_documents(missing)
            for idx, emb in zip(missing_indices, computed):
                text = texts[idx].strip()
                if len(self._cache) >= self._max_cache_size:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[text] = emb
                results[idx] = emb
        return results

    def embed_query(self, text: str) -> list[float]:
        clean_text = text.strip().lower()
        if clean_text in self._cache:
            return self._cache[clean_text]
        emb = self.base.embed_query(text)
        if len(self._cache) >= self._max_cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[clean_text] = emb
        return emb

    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)

    def __getattr__(self, name):
        return getattr(self.base, name)


def _load_embeddings_singleton(model_name: str = "all-MiniLM-L6-v2") -> CachedHuggingFaceEmbeddings:
    global _GLOBAL_EMBEDDINGS
    if _GLOBAL_EMBEDDINGS is None:
        base = HuggingFaceEmbeddings(model_name=model_name)
        _GLOBAL_EMBEDDINGS = CachedHuggingFaceEmbeddings(base)
    return _GLOBAL_EMBEDDINGS


class FAQEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = _load_embeddings_singleton(model_name)

    def split_documents(self, documents, chunk_size=320, chunk_overlap=64):
        # Semantic chunking with 20% overlap across sentence and paragraph boundaries
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""],
            length_function=len,
        )
        return splitter.split_documents(documents)

    def split_documents_sentence_window(self, documents, window_size: int = 1) -> list[Document]:
        """
        Sentence-Window Chunking ('Small-to-Big'):
        Child = 1 sentence (for sharp dense embedding retrieval).
        Metadata['window_context'] = sentence +/- window_size neighbors (60-90 tokens passed to LLM).
        """
        sentence_regex = re.compile(r'(?<=[.!?])\s+')
        window_docs = []

        for doc in documents:
            content = doc.page_content.replace("\r\n", "\n").strip()
            if not content:
                continue

            raw_sentences = [s.strip() for s in sentence_regex.split(content) if s.strip()]
            if not raw_sentences:
                continue

            total_sents = len(raw_sentences)
            for i, sentence in enumerate(raw_sentences):
                # Only index sentences with substantive content
                if len(sentence) < 15 and not any(c.isalnum() for c in sentence):
                    continue

                start = max(0, i - window_size)
                end = min(total_sents, i + window_size + 1)
                window_text = " ".join(raw_sentences[start:end])

                chunk = Document(
                    page_content=sentence,
                    metadata={
                        **doc.metadata,
                        "window_context": window_text,
                        "sentence_index": i,
                        "total_sentences": total_sents,
                    }
                )
                window_docs.append(chunk)

        return window_docs

    def build_index(self, documents, index_path: str):
        if not self.model:
            raise ValueError("Embedding model not loaded")
        if not documents:
            raise ValueError("No documents to index — check your source directories")
        
        import pickle
        
        vectorstore = FAISS.from_documents(documents, self.model)
        vectorstore.save_local(index_path)
        
        # Save raw documents for BM25 fast loading
        docs_path = os.path.join(index_path, "raw_docs.pkl")
        with open(docs_path, "wb") as f:
            pickle.dump(documents, f)
            
        print(f"[EMBEDDER] Indexed {len(documents)} chunks -> {index_path}")

    def process_all_pdfs_texts(self, *source_dirs):
        if len(source_dirs) == 1 and isinstance(source_dirs[0], (list, tuple)):
            dirs = source_dirs[0]
        else:
            dirs = list(source_dirs)
        docs = []
        
        for dir_path in dirs:
            path = Path(dir_path)
            if not path.exists():
                continue
            for file_path in path.glob("**/*.*"):
                loader = None
                
                if file_path.suffix.lower() == '.pdf':
                    loader = PyPDFLoader(str(file_path))
                elif file_path.suffix.lower() == '.txt':
                    loader = TextLoader(str(file_path))
                elif file_path.suffix.lower() == '.md':
                    loader = UnstructuredMarkdownLoader(str(file_path))
                elif file_path.suffix.lower() == '.docx':
                    loader = Docx2txtLoader(str(file_path))
                    
                if loader:
                    try:
                        file_docs = loader.load()
                        for doc in file_docs:
                            doc.metadata["source_file"] = file_path.name
                            ext = file_path.suffix.lower()[1:]
                            doc.metadata["file_type"] = "text" if ext == "txt" else ext
                        docs.extend(file_docs)
                    except Exception as e:
                        print(f"Error loading {file_path.name}: {e}")
                
        return docs
