import pytest
from unittest.mock import patch, MagicMock

from services.rag.retriever import FAQRetriever


@pytest.fixture
def mock_faiss_bm25():
    with patch("services.rag.retriever.FAISS") as mock_faiss, \
         patch("services.rag.retriever.FAQRetriever._load_bm25") as mock_load_bm25, \
         patch("services.rag.retriever._get_embeddings"):
        
        mock_faiss_db = MagicMock()
        mock_faiss.load_local.return_value = mock_faiss_db
        
        mock_bm25 = MagicMock()
        mock_load_bm25.return_value = mock_bm25
        
        yield mock_faiss_db, mock_bm25


def test_search_faq_combined_results(mock_faiss_bm25):
    mock_faiss_db, mock_bm25 = mock_faiss_bm25
    
    # Mock FAISS output
    doc_faiss = MagicMock()
    doc_faiss.page_content = "FAISS Content"
    mock_faiss_db.similarity_search.return_value = [doc_faiss]
    
    # Mock BM25 output
    doc_bm25 = MagicMock()
    doc_bm25.page_content = "BM25 Content"
    mock_bm25.invoke.return_value = [doc_bm25]
    
    retriever = FAQRetriever("fake_path")
    
    result = retriever.search_faq("test query", top_k=2)
    
    assert "FAISS Content" in result
    assert "BM25 Content" in result


def test_search_faq_empty_results(mock_faiss_bm25):
    mock_faiss_db, mock_bm25 = mock_faiss_bm25
    
    mock_faiss_db.similarity_search.return_value = []
    mock_bm25.invoke.return_value = []
    
    retriever = FAQRetriever("fake_path")
    result = retriever.search_faq("random query")
    
    assert result == "I do not have that information in my knowledge base."


@patch("services.rag.retriever._get_embeddings")
@patch("services.rag.retriever.FAISS")
def test_retriever_initialization_failure(mock_faiss, mock_get_emb):
    mock_faiss.load_local.side_effect = Exception("Index missing")
    
    with pytest.raises(Exception, match="Index missing"):
        FAQRetriever("invalid_path")


def test_search_faq_handles_exception(mock_faiss_bm25):
    mock_faiss_db, mock_bm25 = mock_faiss_bm25
    mock_faiss_db.similarity_search.side_effect = Exception("DB error")
    
    retriever = FAQRetriever("fake_path")
    result = retriever.search_faq("test")
    
    assert "Error accessing knowledge base: DB error" in result
