
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

def _build_model_candidates(temperature: float = 0.7):
    """
    Build prioritized list of LLM candidates for fallback resiliency.
    Primary candidate is determined by settings.LLM_PROVIDER.
    Secondary / tertiary candidates act as fallbacks if primary hits 429 rate limit or errors.
    """
    candidates = []

    # Candidate 1: Groq
    groq_llm = None
    if settings.GROQ_API_KEY:
        try:
            from langchain_groq import ChatGroq
            groq_model = settings.LLM_MODEL if settings.LLM_PROVIDER == "groq" else "openai/gpt-oss-20b"
            groq_llm = ChatGroq(
                api_key=settings.GROQ_API_KEY, 
                model=groq_model, 
                temperature=temperature,
                max_retries=1
            )
        except Exception as e:
            print(f"[LLM] Warning: Failed to init ChatGroq candidate: {e}")

    # Candidate 2: Google Gemini
    gemini_llm = None
    if settings.GEMINI_API_KEY:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            gemini_model = settings.LLM_MODEL if settings.LLM_PROVIDER == "google" else "gemini-3.6-flash"
            gemini_llm = ChatGoogleGenerativeAI(
                model=gemini_model,
                google_api_key=settings.GEMINI_API_KEY,
                temperature=temperature
            )
        except Exception as e:
            print(f"[LLM] Warning: Failed to init ChatGoogleGenerativeAI candidate: {e}")

    # Candidate 3: OpenRouter
    openrouter_llm = None
    if settings.OSS_API_KEY:
        try:
            from langchain_openai import ChatOpenAI
            oss_model = settings.LLM_MODEL if settings.LLM_PROVIDER == "openai" else "openai/gpt-4o-mini"
            openrouter_llm = ChatOpenAI(
                api_key=settings.OSS_API_KEY,
                base_url="https://openrouter.ai/api/v1",
                model=oss_model,
                temperature=temperature,
                max_retries=2,
                request_timeout=15.0
            )
        except Exception as e:
            print(f"[LLM] Warning: Failed to init ChatOpenAI candidate: {e}")

    # Order candidates according to settings.LLM_PROVIDER
    if settings.LLM_PROVIDER == "groq":
        for m in [groq_llm, gemini_llm, openrouter_llm]:
            if m is not None:
                candidates.append(m)
    elif settings.LLM_PROVIDER == "google":
        for m in [gemini_llm, groq_llm, openrouter_llm]:
            if m is not None:
                candidates.append(m)
    elif settings.LLM_PROVIDER == "openai":
        for m in [openrouter_llm, gemini_llm, groq_llm]:
            if m is not None:
                candidates.append(m)
    else:
        for m in [groq_llm, gemini_llm, openrouter_llm]:
            if m is not None:
                candidates.append(m)

    if not candidates:
        raise RuntimeError("No LLM providers available! Check GROQ_API_KEY, GEMINI_API_KEY, or OSS_API_KEY.")

    return candidates

def _build_llm(temperature: float = 0.7):
    candidates = _build_model_candidates(temperature=temperature)
    if len(candidates) == 1:
        return candidates[0]
    return candidates[0].with_fallbacks(candidates[1:])

_llm_with_tools = None
_guardrail_llm = None

def get_llm_with_tools():
    '''Build the configured LLM and bind the current tool list with multi-tier fallback support.'''
    global _llm_with_tools
    if _llm_with_tools is not None:
        return _llm_with_tools

    candidates = _build_model_candidates()
    bound_candidates = [m.bind_tools(tools).with_config({"run_name": "agent_model"}) for m in candidates]

    if len(bound_candidates) == 1:
        _llm_with_tools = bound_candidates[0]
    else:
        _llm_with_tools = bound_candidates[0].with_fallbacks(bound_candidates[1:]).with_config({"run_name": "agent_model"})

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

    # Normalize response.content if returned as a list of dicts (e.g. Gemini multimodal/text blocks)
    if isinstance(response.content, list) and not getattr(response, "tool_calls", None):
        text_parts = []
        for part in response.content:
            if isinstance(part, dict) and "text" in part:
                text_parts.append(str(part["text"]))
            elif isinstance(part, str):
                text_parts.append(part)
        if text_parts:
            response.content = "".join(text_parts)

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
        raw_content = response.content
        if isinstance(raw_content, list):
            text_parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in raw_content]
            content = "".join(text_parts).upper()
        else:
            content = str(raw_content).upper()

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
