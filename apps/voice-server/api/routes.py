from fastapi import APIRouter, WebSocket, HTTPException, Security, Request
from fastapi.responses import PlainTextResponse, Response
from fastapi.security import APIKeyHeader
from core.config import settings

router = APIRouter()
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

def get_api_key(api_key: str = Security(api_key_header)):
    if not api_key:
        raise HTTPException(status_code=403, detail="No token provided")
    if api_key != f"Bearer {settings.TWILIO_AUTH_TOKEN}" and api_key != settings.TWILIO_AUTH_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")
    return api_key

@router.get("/warmup-status")
def warmup_status():
    from main import _startup_status
    critical_ready = _startup_status["db"] and _startup_status["langgraph"]
    return {"ready": critical_ready, "components": _startup_status}

@router.get("/dev-token")
def dev_token():
    """
    Returns the main Twilio token as our 'dev token', and generates a fake stream ID for testing.
    
    Returns:
        dict: A dictionary containing the token and a generated stream_sid.
    """
    import uuid
    return {
        "token": settings.TWILIO_AUTH_TOKEN,
        "stream_sid": f"DEV_STREAM_{uuid.uuid4().hex[:8]}"
    }

@router.post("/twilio/twiml")
async def twilio_twiml(request: Request):
    """
    Standard Twilio Media Stream TwiML endpoint.
    
    Args:
        request (Request): The incoming request object.
        
    Returns:
        PlainTextResponse: The TwiML XML response for Twilio to connect to the websocket.
    """
    host = settings.BASE_URL
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="wss://{host}/ws">
            <Parameter name="token" value="{settings.TWILIO_AUTH_TOKEN}" />
        </Stream>
    </Connect>
</Response>"""
    return PlainTextResponse(xml, media_type="text/xml")

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for handling Twilio Media Streams.
    
    Args:
        websocket (WebSocket): The WebSocket connection instance.
    """
    await websocket.accept()
    
    auth = websocket.headers.get("Authorization")
    token = websocket.query_params.get("token")
    
    is_valid = False
    if auth and auth == f"Bearer {settings.TWILIO_AUTH_TOKEN}":
        is_valid = True
    elif token and token == settings.TWILIO_AUTH_TOKEN:
        is_valid = True
        
    if not is_valid:
        await websocket.close(code=1008)
        return
        
    import uuid
    call_sid = websocket.query_params.get("stream_sid") or f"WEB_{uuid.uuid4().hex[:8]}"
    from services.pipecat_pipeline.bot import run_bot
    await run_bot(websocket, call_sid)

@router.get("/twiml/transfer/{destination}")
async def twiml_transfer(destination: str):
    phone_map = {
        "emergency": settings.EMERGENCY_PHONE,
        "front_desk": settings.FRONT_DESK_PHONE,
    }
    phone = phone_map.get(destination, settings.FRONT_DESK_PHONE)
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response><Dial>{phone}</Dial></Response>"""
    return Response(content=twiml, media_type="application/xml")
