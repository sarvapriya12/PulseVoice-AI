import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from services.langgraph_agent.nodes import should_continue, call_model


def test_should_continue_with_tools():
    # Mock message with tool calls
    last_message = AIMessage(content="", tool_calls=[{"id": "call_123", "name": "check_availability", "args": {}}])
    state = {"messages": [last_message]}
    
    assert should_continue(state) == "tools"


def test_should_continue_without_tools():
    last_message = AIMessage(content="Hello there")
    state = {"messages": [last_message]}
    
    assert should_continue(state) == "end"


@pytest.mark.asyncio
@patch("services.langgraph_agent.nodes.ChatOpenAI")
async def test_call_model_prepends_system_prompt(mock_chat_openai):
    mock_llm_instance = MagicMock()
    mock_chat_openai.return_value = mock_llm_instance
    
    # Mock bind_tools
    mock_bound_llm = AsyncMock()
    mock_llm_instance.bind_tools.return_value = mock_bound_llm
    
    # Mock LLM response
    mock_response = AIMessage(content="Response")
    mock_bound_llm.ainvoke.return_value = mock_response
    
    state = {"messages": [HumanMessage(content="Hi")]}
    
    result = await call_model(state)
    
    # Assert result
    assert result == {"messages": [mock_response]}
    
    # Assert ainvoke was called with SystemMessage prepended
    called_messages = mock_bound_llm.ainvoke.call_args[0][0]
    assert len(called_messages) == 2
    assert isinstance(called_messages[0], SystemMessage)
    assert isinstance(called_messages[1], HumanMessage)


@pytest.mark.asyncio
@patch("services.langgraph_agent.nodes.ChatOpenAI")
async def test_call_model_no_duplicate_system_prompt(mock_chat_openai):
    mock_llm_instance = MagicMock()
    mock_chat_openai.return_value = mock_llm_instance
    
    mock_bound_llm = AsyncMock()
    mock_llm_instance.bind_tools.return_value = mock_bound_llm
    mock_bound_llm.ainvoke.return_value = AIMessage(content="Response")
    
    state = {"messages": [SystemMessage(content="Existing system prompt"), HumanMessage(content="Hi")]}
    
    await call_model(state)
    
    called_messages = mock_bound_llm.ainvoke.call_args[0][0]
    assert len(called_messages) == 2
    assert called_messages[0].content == "Existing system prompt"
