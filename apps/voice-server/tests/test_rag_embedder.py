import pytest
from unittest.mock import patch, MagicMock

from services.rag.embedder import FAQEmbedder


@pytest.fixture
def mock_embedder():
    with patch("services.rag.embedder.FAQEmbedder._load_model"):
        yield FAQEmbedder("mock-model")


def test_split_documents(mock_embedder):
    # Mock documents
    doc1 = MagicMock()
    doc1.page_content = "A" * 1000
    doc1.metadata = {}
    
    chunks = mock_embedder.split_documents([doc1], chunk_size=600, chunk_overlap=100)
    
    # Should be at least 2 chunks
    assert len(chunks) >= 2
    # Check max size of a chunk
    assert len(chunks[0].page_content) <= 600


def test_build_index_no_model(mock_embedder):
    mock_embedder.model = None
    
    with pytest.raises(ValueError, match="Model not loaded"):
        mock_embedder.build_index([], "dummy_path")


@patch("services.rag.embedder.FAISS")
def test_build_index_success(mock_faiss, mock_embedder):
    mock_embedder.model = MagicMock()
    mock_faiss_instance = MagicMock()
    mock_faiss.from_documents.return_value = mock_faiss_instance
    
    mock_embedder.build_index(["dummy_doc"], "test_path")
    
    mock_faiss.from_documents.assert_called_once_with(["dummy_doc"], mock_embedder.model)
    mock_faiss_instance.save_local.assert_called_once_with("test_path")


@patch("services.rag.embedder.Path.exists")
@patch("services.rag.embedder.Path.glob")
@patch("services.rag.embedder.TextLoader")
def test_process_all_pdfs_texts(mock_text_loader, mock_glob, mock_exists, mock_embedder):
    mock_exists.return_value = True
    mock_path = MagicMock()
    mock_path.suffix = ".txt"
    mock_path.name = "test.txt"
    mock_glob.return_value = [mock_path]
    
    mock_loader_instance = MagicMock()
    mock_doc = MagicMock()
    mock_doc.metadata = {}
    mock_loader_instance.load.return_value = [mock_doc]
    mock_text_loader.return_value = mock_loader_instance
    
    docs = mock_embedder.process_all_pdfs_texts("pdf_dir", "text_dir")
    
    assert len(docs) > 0
    assert docs[0].metadata["source_file"] == "test.txt"
    assert docs[0].metadata["file_type"] == "text"
