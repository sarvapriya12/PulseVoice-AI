from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from langgraph.prebuilt import ToolNode

from core.config import settings
from services.langgraph_agent.state import AgentState
from services.langgraph_agent.nodes import call_model, should_continue, tools, guardrail_node, refusal_node

def route_after_guardrail(state: AgentState) -> str:
    return "refusal" if state.get("intent") == "off_topic" else "agent"

_workflow = StateGraph(AgentState)
_workflow.add_node("guardrail", guardrail_node)
_workflow.add_node("agent", call_model)
_workflow.add_node("tools", ToolNode(tools))
_workflow.add_node("refusal", refusal_node)

_workflow.add_edge(START, "guardrail")
_workflow.add_conditional_edges("guardrail", route_after_guardrail, {"agent": "agent", "refusal": "refusal"})
_workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
_workflow.add_edge("tools", "agent")
_workflow.add_edge("refusal", END)

async def build_graph():
    if "sqlite" in settings.DATABASE_URL:
        checkpointer = MemorySaver()
        pool = None
    else:
        db_url = settings.DATABASE_URL.replace("+psycopg2", "")
        pool = AsyncConnectionPool(conninfo=db_url, min_size=2, max_size=10)
        await pool.open()
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        
    app = _workflow.compile(checkpointer=checkpointer)
    app._pool = pool
    return app
