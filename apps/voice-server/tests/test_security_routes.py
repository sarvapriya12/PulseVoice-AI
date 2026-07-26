import base64
import hmac
import hashlib
from core.config import settings

def generate_valid_twilio_signature(url: str, params: dict, auth_token: str) -> str:
    '''Mocks the exact HMAC-SHA1 signature Twilio uses for webhooks.'''
    # Twilio appends POST params alphabetically to the URL
    data = url
    for k, v in sorted(params.items()):
        data += f"{k}{v}"
        
    mac = hmac.new(auth_token.encode("utf-8"), data.encode("utf-8"), hashlib.sha1)
    return base64.b64encode(mac.digest()).decode("utf-8")

def test_webhook_rejects_unsigned_requests(client):
    '''SECURITY TEST: The webhook MUST reject requests without a valid Twilio Signature.'''
    response = client.post("/twilio/twiml", data={"CallSid": "CA12345"})
    assert response.status_code == 403
    assert "Invalid Twilio Signature" in response.json()["detail"]

def test_webhook_rejects_fake_signatures(client):
    '''SECURITY TEST: The webhook MUST reject requests with invalid/spoofed signatures.'''
    headers = {"X-Twilio-Signature": "FakeSignature123"}
    response = client.post("/twilio/twiml", headers=headers, data={"CallSid": "CA12345"})
    assert response.status_code == 403

def test_webhook_accepts_valid_signature_and_issues_token(client):
    '''INTEGRATION TEST: Valid Twilio signatures should return XML with a WebSocket token.'''
    # We must match the EXACT url the TestClient uses
    url = "http://testserver/twilio/twiml"
    params = {"CallSid": "CA1234567890"}
    
    # Generate the valid signature using our ACTUAL test config token
    valid_sig = generate_valid_twilio_signature(url, params, settings.TWILIO_AUTH_TOKEN)
    
    headers = {"X-Twilio-Signature": valid_sig}
    response = client.post("/twilio/twiml", headers=headers, data=params)
    
    assert response.status_code == 200
    assert "text/xml" in response.headers["content-type"]
    
    # Verify the TwiML tells Twilio to connect to our WebSocket and passes a Token
    twiml = response.text
    assert "<Response>" in twiml
    assert "<Connect>" in twiml
    assert "<Stream url=" in twiml
    assert "token=" in twiml
