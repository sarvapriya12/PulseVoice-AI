from twilio.rest import Client
from core.config import settings
from langchain_core.tools import tool

class TwilioRepository:
    def __init__(self, client: Client):
        self.client = client

    def send_sms(self, body: str, to: str) -> str:
        message = self.client.messages.create(
            body=body,
            from_=settings.TWILIO_PHONE_NUMBER,
            to=to
        )
        return message.sid

    def redirect_call(self, call_sid: str, twiml: str):
        """
        Redirects an ongoing Twilio call using a TwiML string.
        
        Args:
            call_sid (str): The unique identifier for the Twilio call.
            twiml (str): The TwiML XML string to execute.
        """
        self.client.calls(call_sid).update(twiml=twiml)


class TwilioService:
    def __init__(self, repo: TwilioRepository):
        self.repo = repo

    async def text_patient(self, phone_number: str, message: str) -> str:
        import asyncio
        try:
            sid = await asyncio.to_thread(self.repo.send_sms, body=message, to=phone_number)
            return f"Successfully sent SMS to {phone_number}. SID: {sid}"
        except Exception as e:
            return f"Error sending SMS: {str(e)}"

    async def text_waitlist(self, patient_name: str, date_str: str) -> str:
        import asyncio
        try:
            body = f"Alert: {patient_name} wants to be on the waitlist for {date_str}."
            sid = await asyncio.to_thread(self.repo.send_sms, body=body, to=settings.WAITLIST_PHONE_NUMBER)
            return f"Successfully sent waitlist alert for {patient_name}. SID: {sid}"
        except Exception as e:
            return f"Error sending waitlist SMS: {str(e)}"

    async def transfer_call(self, call_sid: str, target: str) -> str:
        import asyncio
        try:
            if target == "emergency":
                phone = settings.EMERGENCY_PHONE
                target_name = "emergency line"
            else:
                phone = settings.FRONT_DESK_PHONE
                target_name = "front desk"
                
            twiml = f"<Response><Dial>{phone}</Dial></Response>"
            await asyncio.to_thread(self.repo.redirect_call, call_sid, twiml)
            return f"Successfully transferred call {call_sid} to {target_name} ({phone})."
        except Exception as e:
            return f"Error transferring call: {str(e)}"


# Factory function to create LangChain tools
def make_twilio_tools(service: TwilioService):
    from langchain_core.runnables import RunnableConfig
    
    @tool
    async def text_patient(phone_number: str, message: str) -> str:
        """Send an SMS text message to a patient."""
        return await service.text_patient(phone_number, message)
        
    @tool
    async def text_waitlist(patient_name: str, date_str: str) -> str:
        """Alert the waitlist manager via SMS that a patient wants to be waitlisted."""
        return await service.text_waitlist(patient_name, date_str)
        
    @tool
    async def transfer_call(target: str, config: RunnableConfig) -> str:
        """Transfer the ongoing phone call. Target must be 'front_desk' or 'emergency'."""
        if target not in ("front_desk", "emergency"):
            return "Invalid transfer target. Must be 'front_desk' or 'emergency'."
        call_sid = config["configurable"].get("thread_id")
        if not call_sid:
            return "Error: No active call context found."
        return await service.transfer_call(call_sid, target)
        
    return [text_patient, text_waitlist, transfer_call]
