import os
import sys

# Add the root directory to the python path so it can import 'services'
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

# Force HuggingFace models (Embeddings) to cache strictly in local_model folder
hf_cache_dir = os.path.join(base_dir, "local_model", "huggingface_cache")
os.makedirs(hf_cache_dir, exist_ok=True)
os.environ["HF_HOME"] = hf_cache_dir
os.environ["HF_HUB_CACHE"] = hf_cache_dir

from services.rag.embedder import FAQEmbedder

def main():
    print("Initializing FAISS Embedder...")
    embedder = FAQEmbedder()
    
    docs_dir = os.path.join(base_dir, "data", "documents")
    index_dir = os.path.join(base_dir, "data", "faiss_index")
    
    print(f"Loading documents from {docs_dir}...")
    raw_docs = embedder.process_all_pdfs_texts([docs_dir])
    
    if not raw_docs:
        print("⚠️ No documents found in data/documents. Add some PDFs or TXT files and run again.")
        return
        
    # Extract unique file names from metadata
    loaded_files = set([doc.metadata.get("source_file", "Unknown") for doc in raw_docs])
    print(f"\nSuccessfully loaded {len(loaded_files)} document(s):")
    for file_name in loaded_files:
        print(f"  - {file_name}")
    print()
        
    print(f"Splitting {len(raw_docs)} document pages/sections into sentence-window chunks...")
    chunks = embedder.split_documents_sentence_window(raw_docs, window_size=1)
    
    print(f"Building FAISS index with {len(chunks)} chunks...")
    embedder.build_index(chunks, index_path=index_dir)
    print(f"[SUCCESS] RAG Index successfully built and saved to {index_dir}!")

if __name__ == "__main__":
    main()
