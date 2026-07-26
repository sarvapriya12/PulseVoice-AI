
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from .state import AgentState
from tools.db_tools import AppointmentRepository, AppointmentService
from core.database import AsyncSessionLocal
from twilio.rest import Client
from core.config import settings
from tools.external_tools import make_twilio_tools, TwilioRepository, TwilioService
from tools.rag_tools import make_rag_tools
from services.rag.retriever import FAQRetriever

# 1. Initialize the global RAG Retriever gracefully
try:
    rag_retriever = FAQRetriever()
    rag_tools = make_rag_tools(rag_retriever)
except Exception as e:
    print(f"Warning: RAG Retriever failed to initialize ({e}). RAG tools disabled.")
    rag_tools = []

# 2. Wrap our DB methods into LangChain `@tool` functions!
from langchain_core.runnables import RunnableConfig

@tool
async def check_availability(date_str: str, config: RunnableConfig) -> str:
    """
    Check which appointment slots are open on a given date.
    ALWAYS call this before offering any time slot to a patient.
    date_str format: YYYY-MM-DD (e.g. '2025-08-15').
    """
    cache = config.get("configurable", {}).get("db_cache", {})
    async with AsyncSessionLocal() as db:
        repo = AppointmentRepository(db)
        service = AppointmentService(repo)
        return await service.check_availability(date_str, cache)

@tool
async def book_appointment(patient_name: str, patient_phone: str, patient_age: int, time_str: str, config: RunnableConfig) -> str:
    """
    Book a confirmed appointment slot for a patient.
    Only call this AFTER check_availability confirms the slot is free and the patient has agreed to it.
    time_str format: YYYY-MM-DD HH:MM:SS (e.g. '2025-08-15 10:00:00').
    """
    cache = config.get("configurable", {}).get("db_cache", {})
    async with AsyncSessionLocal() as db:
        repo = AppointmentRepository(db)
        service = AppointmentService(repo)
        return await service.book_appointment(patient_name, patient_phone, patient_age, time_str, cache)

@tool
async def cancel_appointment(patient_phone: str, date_str: str, time_str: str = None) -> str:
    """
    Cancel a patient's appointment by their phone number and the date of the appointment.
    Use the caller's phone number. date_str format: YYYY-MM-DD.
    If there are multiple appointments on that date, the system will prompt you to provide the time_str (HH:MM:SS).
    """
    async with AsyncSessionLocal() as db:
        repo = AppointmentRepository(db)
        service = AppointmentService(repo)
        return await service.cancel_appointment_by_phone(patient_phone, date_str, time_str)

@tool
async def add_to_waitlist(patient_name: str, patient_phone: str, patient_age: int, date_str: str) -> str:
    '''Add a patient to the automated database waitlist for a specific date (YYYY-MM-DD).'''
    async with AsyncSessionLocal() as db:
        repo = AppointmentRepository(db)
        service = AppointmentService(repo)
        return await service.add_to_waitlist(patient_name, patient_phone, patient_age, date_str)

@tool
async def check_waitlist(date_str: str) -> str:
    '''Check if anyone is on the waitlist for a specific date (YYYY-MM-DD).'''
    async with AsyncSessionLocal() as db:
        repo = AppointmentRepository(db)
        service = AppointmentService(repo)
        return await service.check_waitlist(date_str)

@tool
async def take_message(patient_name: str, patient_phone: str, message: str) -> str:
    '''ONLY call this tool if the patient explicitly asks to leave a message for the doctor, or if they have a highly complex medical condition you cannot safely provide general information for.'''
    async with AsyncSessionLocal() as db:
        repo = AppointmentRepository(db)
        service = AppointmentService(repo)
        return await service.take_message(patient_name, patient_phone, message)

@tool
async def clear_memory() -> str:
    '''Use this tool when the patient asks to start fresh, forget previous context, or clear the conversation history.'''
    return "__RESET_CONTEXT_SIGNAL__"

# 3. Initialize Twilio tools gracefully
try:
    if settings.TWILIO_ACCOUNT_SID and settings.TWILIO_AUTH_TOKEN:
        twilio_client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        twilio_tools = make_twilio_tools(TwilioService(TwilioRepository(twilio_client)))
    else:
        print("Warning: Twilio credentials not set. Twilio tools disabled.")
        twilio_tools = []
except Exception as e:
    print(f"Warning: Twilio client failed to initialize ({e}). Twilio tools disabled.")
    twilio_tools = []

# 4. Bundle them for the LLM and the ToolNode
# We include the DB tools directly, and the RAG/Twilio tools from their factories
tools = [
    book_appointment, 
    check_availability, 
    cancel_appointment, 
    add_to_waitlist, 
    check_waitlist,
    take_message,
    clear_memory
] + rag_tools + twilio_tools

def _build_llm(temperature: float = 0.7):
    if settings.LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(api_key=settings.GROQ_API_KEY, model=settings.LLM_MODEL, temperature=temperature)
    return ChatOpenAI(api_key=settings.OSS_API_KEY, base_url="https://openrouter.ai/api/v1", model=settings.LLM_MODEL, temperature=temperature)

_llm_with_tools = None
_guardrail_llm = None

def get_llm_with_tools():
    '''Build the configured LLM and bind the current tool list.'''
    global _llm_with_tools
    if _llm_with_tools is not None:
        return _llm_with_tools
        
    llm = _build_llm()
    _llm_with_tools = llm.bind_tools(tools).with_config({"run_name": "agent_model"})
    return _llm_with_tools


from services.pipecat_pipeline.prompts import SYSTEM_PROMPT

async def call_model(state: AgentState) -> dict:
    """
    Invoke the LLM with the current conversation history.

    The LLM will either:
      a) Respond with a plain AIMessage  → graph routes to END.
      b) Respond with tool_calls        → graph routes to tool_node.
      
    This function prepends the system prompt if the conversation doesn't have one yet,
    preventing the system prompt from being duplicated on every turn.
    """
    from langchain_core.messages import SystemMessage

    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        from datetime import datetime
        current_date_str = datetime.now().strftime("%A, %B %d, %Y")
        dynamic_prompt = SYSTEM_PROMPT + f"\n\nCURRENT DATE: {current_date_str}. All appointments must be booked in the future."
        messages = [SystemMessage(content=dynamic_prompt)] + list(messages)

    llm_with_tools = get_llm_with_tools()
    response = await llm_with_tools.ainvoke(messages)

    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """
    Decide the next node based on the LLM's last response.
    """
    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "tools"

    return "end"


async def guardrail_node(state: AgentState) -> dict:
    """
    Evaluate the last user message and classify it as on_topic or off_topic.
    
    We use the same LLM configuration but with temperature=0.0 for classification.
    """
    from langchain_core.messages import SystemMessage
    
    last_message = state["messages"][-1]
    if getattr(last_message, "type", "") != "human":
        return {}

    prompt = "Classify the latest user input given the context of the conversation. Is it related to a clinic, healthcare, doctors, medical issues, appointments, or asking for help with these? It may also be a simple greeting. Answer ONLY with 'YES' if it is appropriate for a clinic assistant, or 'NO' if it is off-topic (e.g. trivia, tech, celebrities, etc.)."
    
    global _guardrail_llm
    if _guardrail_llm is None:
        _guardrail_llm = _build_llm(temperature=0.0).with_config({"run_name": "guardrail_model"})
        
    try:
        invoke_messages = [SystemMessage(content=prompt)] + state["messages"][-4:]
        response = await _guardrail_llm.ainvoke(invoke_messages)
        content = response.content.upper()
        if "NO" in content and "YES" not in content:
            print(f"[GUARDRAIL] Intercepted off-topic message: {last_message.content}")
            return {"intent": "off_topic"}
    except Exception as e:
        print(f"[GUARDRAIL] Error during classification: {e}")
        
    return {"intent": "on_topic"}


async def refusal_node(state: AgentState) -> dict:
    '''Return a canned refusal message for off-topic inputs.'''
    from langchain_core.messages import AIMessage
    msg = AIMessage(content="I am a medical assistant for Dr. Smith's clinic and cannot help with that. Is there anything health or clinic-related I can assist you with?")
    return {"messages": [msg]}
