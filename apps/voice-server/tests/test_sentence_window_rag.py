import pytest
from langchain_core.documents import Document
from services.rag.embedder import FAQEmbedder

def test_sentence_window_chunking():
    embedder = FAQEmbedder()
    
    sample_text = (
        "Austin Health Clinic welcomes new patients. "
        "Please arrive 15 minutes before your scheduled appointment. "
        "Bring a photo ID and your current insurance card. "
        "Failure to bring documents may delay your check-in process. "
        "We look forward to serving your healthcare needs."
    )
    
    input_docs = [Document(page_content=sample_text, metadata={"source_file": "intake_sample.txt"})]
    
    window_chunks = embedder.split_documents_sentence_window(input_docs, window_size=1)
    
    assert len(window_chunks) == 5
    
    # Check middle chunk (sentence index 2: "Bring a photo ID and your current insurance card.")
    middle_chunk = window_chunks[2]
    assert middle_chunk.page_content == "Bring a photo ID and your current insurance card."
    assert middle_chunk.metadata["sentence_index"] == 2
    assert middle_chunk.metadata["source_file"] == "intake_sample.txt"
    
    # Check that window_context contains previous and next sentences
    window_ctx = middle_chunk.metadata["window_context"]
    assert "Please arrive 15 minutes before" in window_ctx
    assert "Bring a photo ID" in window_ctx
    assert "Failure to bring documents" in window_ctx
