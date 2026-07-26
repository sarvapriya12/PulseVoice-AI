import os
from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil

from core.config import settings
router = APIRouter(prefix="/api/admin", tags=["Admin"])
DOCS_DIR = settings.DOCS_DIR

# Ensure directory exists
os.makedirs(DOCS_DIR, exist_ok=True)

@router.get("/docs")
async def list_docs():
    """List all FAQ PDF documents currently stored in the system."""
    try:
        files = os.listdir(DOCS_DIR)
        valid_extensions = ('.pdf', '.txt', '.md', '.docx')
        docs = [f for f in files if f.lower().endswith(valid_extensions)]
        return {"documents": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/docs")
async def upload_doc(file: UploadFile = File(...)):
    """Upload a new document to the FAQ knowledge base."""
    valid_extensions = ('.pdf', '.txt', '.md', '.docx')
    if not file.filename.lower().endswith(valid_extensions):
        raise HTTPException(status_code=400, detail="Only PDF, TXT, MD, and DOCX files are allowed.")
        
    safe_filename = os.path.basename(file.filename)
    file_path = os.path.join(DOCS_DIR, safe_filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"message": f"Successfully uploaded {file.filename}", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/docs/{filename}")
async def delete_doc(filename: str):
    """Delete a specific PDF from the knowledge base."""
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(DOCS_DIR, safe_filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    try:
        os.remove(file_path)
        return {"message": f"Successfully deleted {filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rebuild-index")
async def rebuild_index():
    """
    Rebuild the FAISS index from the current PDFs and hot-swap the 
    LangGraph retriever's memory without restarting the server!
    
    This function runs the heavy embedding process in a background thread 
    so it doesn't freeze the API or WebSocket, and then triggers an instant 
    hot-reload on the global retriever singleton.
    
    Returns:
        dict: A message confirming successful rebuild and hot-swap.
        
    Raises:
        HTTPException: If the RAG Retriever is not loaded in memory or an error occurs.
    """
    import asyncio
    from services.rag.embedder import FAQEmbedder
    from services.langgraph_agent.nodes import rag_retriever
    
    try:
        def _build():
            embedder = FAQEmbedder()
            raw_docs = embedder.process_all_pdfs_texts([DOCS_DIR])
            if not raw_docs:
                raise ValueError("No documents found in the directory.")
            chunks = embedder.split_documents(raw_docs)
            embedder.build_index(chunks, index_path="data/faiss_index")
            
        await asyncio.to_thread(_build)
        
        if rag_retriever:
            rag_retriever.reload_index()
            return {"message": "Index successfully rebuilt and RAG memory hot-swapped!"}
        else:
            raise HTTPException(status_code=500, detail="RAG Retriever is not loaded in memory.")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rebuild index: {str(e)}")
