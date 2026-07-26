import pytest
from unittest.mock import AsyncMock
from tools.external_tools import TwilioService, TwilioRepository
from core.config import settings

@pytest.fixture
def mock_repo():
    repo = AsyncMock(spec=TwilioRepository)
    # Give default returns for the methods
    repo.send_sms.return_value = "mock_sid_12345"
    repo.redirect_call.return_value = None
    return repo

@pytest.fixture
def twilio_service(mock_repo):
    return TwilioService(repo=mock_repo)

@pytest.mark.asyncio
async def test_text_patient_success(twilio_service, mock_repo):
    phone_number = "+15551234567"
    message = "Hello, your appointment is confirmed."
    
    response = await twilio_service.text_patient(phone_number, message)
    
    mock_repo.send_sms.assert_called_once_with(body=message, to=phone_number)
    assert "mock_sid_12345" in response
    assert phone_number in response

@pytest.mark.asyncio
async def test_text_patient_error(twilio_service, mock_repo):
    phone_number = "+15551234567"
    message = "Hello"
    
    mock_repo.send_sms.side_effect = Exception("Twilio API Error")
    
    response = await twilio_service.text_patient(phone_number, message)
    
    assert "Error sending SMS" in response
    assert "Twilio API Error" in response

@pytest.mark.asyncio
async def test_text_waitlist_success(twilio_service, mock_repo):
    patient_name = "John Doe"
    date = "2023-12-01"
    
    response = await twilio_service.text_waitlist(patient_name, date)
    
    mock_repo.send_sms.assert_called_once()
    call_kwargs = mock_repo.send_sms.call_args.kwargs
    assert call_kwargs["to"] == settings.WAITLIST_PHONE_NUMBER
    assert patient_name in call_kwargs["body"]
    assert date in call_kwargs["body"]
    
    assert "mock_sid_12345" in response
    assert patient_name in response

@pytest.mark.asyncio
async def test_text_waitlist_error(twilio_service, mock_repo):
    mock_repo.send_sms.side_effect = Exception("Twilio Alert Error")
    
    response = await twilio_service.text_waitlist("John", "2023-12-01")
    
    assert "Error sending waitlist SMS" in response
    assert "Twilio Alert Error" in response

@pytest.mark.asyncio
async def test_transfer_call_front_desk(twilio_service, mock_repo):
    call_sid = "call_123"
    
    response = await twilio_service.transfer_call(call_sid, target="front_desk")
    
    expected_twiml = f"<Response><Dial>{settings.FRONT_DESK_PHONE}</Dial></Response>"
    mock_repo.redirect_call.assert_called_once_with(call_sid, expected_twiml)
    
    assert "front desk" in response
    assert settings.FRONT_DESK_PHONE in response
    assert call_sid in response

@pytest.mark.asyncio
async def test_transfer_call_emergency(twilio_service, mock_repo):
    call_sid = "call_456"
    
    response = await twilio_service.transfer_call(call_sid, target="emergency")
    
    expected_twiml = f"<Response><Dial>{settings.EMERGENCY_PHONE}</Dial></Response>"
    mock_repo.redirect_call.assert_called_once_with(call_sid, expected_twiml)
    
    assert "emergency line" in response
    assert settings.EMERGENCY_PHONE in response
    assert call_sid in response

@pytest.mark.asyncio
async def test_transfer_call_error(twilio_service, mock_repo):
    call_sid = "call_err"
    mock_repo.redirect_call.side_effect = Exception("Redirect Failed")
    
    response = await twilio_service.transfer_call(call_sid, target="front_desk")
    
    assert "Error transferring call" in response
    assert "Redirect Failed" in response
