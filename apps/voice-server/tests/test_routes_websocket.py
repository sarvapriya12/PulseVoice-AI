import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
TestClient.__test__ = False
from fastapi import WebSocketDisconnect
from core.config import settings

def test_websocket_no_token(client: TestClient):
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/ws"):
            pass
    assert exc_info.value.code == 1008

def test_websocket_invalid_token(client: TestClient):
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/ws?token=invalid_token"):
            pass
    assert exc_info.value.code == 1008

@patch("api.routes.run_bot", new_callable=AsyncMock)
def test_websocket_valid_token(mock_run_bot, client: TestClient):
    # Connect with the valid token from settings
    valid_token = settings.TWILIO_AUTH_TOKEN
    with client.websocket_connect(f"/ws?token={valid_token}"):
        # If it doesn't raise WebSocketDisconnect(1008), connection succeeded
        pass
    
    # run_bot should have been called
    mock_run_bot.assert_called_once()
