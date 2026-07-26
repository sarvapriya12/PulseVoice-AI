import asyncio
from langchain_core.tools import tool
from services.rag.retriever import FAQRetriever

def make_rag_tools(retriever: FAQRetriever):
    @tool
    async def search_faq(query: str) -> str:
        """Search the clinic's knowledge base / FAQ for information (e.g. policies, procedures, general info)."""
        return await asyncio.to_thread(retriever.search_faq, query)

    return [search_faq]
