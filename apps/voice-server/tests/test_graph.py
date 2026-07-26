import pytest
from unittest.mock import patch, AsyncMock
from langgraph.checkpoint.memory import MemorySaver


@pytest.mark.asyncio
@patch("services.langgraph_agent.graph.settings")
@patch("services.langgraph_agent.graph.workflow")
async def test_build_graph_sqlite(mock_workflow, mock_settings):
    mock_settings.DATABASE_URL = "sqlite:///./test.db"
    
    from services.langgraph_agent.graph import build_graph
    
    await build_graph()
    
    # Assert workflow.compile was called with MemorySaver
    kwargs = mock_workflow.compile.call_args[1]
    assert isinstance(kwargs["checkpointer"], MemorySaver)


@pytest.mark.asyncio
@patch("services.langgraph_agent.graph.settings")
@patch("services.langgraph_agent.graph.workflow")
@patch("services.langgraph_agent.graph.AsyncConnectionPool")
@patch("services.langgraph_agent.graph.AsyncPostgresSaver")
async def test_build_graph_postgres(mock_saver, mock_pool, mock_workflow, mock_settings):
    mock_settings.DATABASE_URL = "postgresql+psycopg2://user:pass@localhost/db"
    
    mock_saver_instance = AsyncMock()
    mock_saver.return_value = mock_saver_instance
    
    from services.langgraph_agent.graph import build_graph
    
    await build_graph()
    
    # Check AsyncConnectionPool was called with formatted URL
    mock_pool.assert_called_once_with("postgresql://user:pass@localhost/db")
    
    # Check setup was called on the saver
    mock_saver_instance.setup.assert_called_once()
    
    # Check workflow.compile was called with the PostgresSaver
    kwargs = mock_workflow.compile.call_args[1]
    assert kwargs["checkpointer"] == mock_saver_instance
