import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class FAQEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            self.model = HuggingFaceEmbeddings(model_name=self.model_name)
        except Exception as e:
            print(f"Error loading embedding model: {e}")

    def split_documents(self, documents, chunk_size=400, chunk_overlap=50):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return splitter.split_documents(documents)

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
            
        print(f"[EMBEDDER] Indexed {len(documents)} chunks → {index_path}")

    def process_all_pdfs_texts(self, source_dirs: list[str]):
        docs = []
        
        for dir_path in source_dirs:
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
                            doc.metadata["file_type"] = file_path.suffix.lower()[1:]
                        docs.extend(file_docs)
                    except Exception as e:
                        print(f"Error loading {file_path.name}: {e}")
                
        return docs
