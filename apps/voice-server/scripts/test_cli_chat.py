import os
import sys
import asyncio

# Add the root directory to the python path so it can import 'services'
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from services.langgraph_agent.graph import build_graph
from langchain_core.messages import HumanMessage
import uuid

async def main():
    print("Initializing Clinic Voice Agent (Text Mode)...")
    app = await build_graph()
    
    # Generate a random session ID to track conversation history
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"Session {thread_id} Started.")
    print("=" * 50)
    print("Welcome to the Clinic Bot! Type 'exit' or 'quit' to close.")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            if not user_input.strip():
                continue
                
            state = {"messages": [HumanMessage(content=user_input)]}
            
            print("Bot is thinking...")
            result = await app.ainvoke(state, config)
            
            # Get the last message in the state which should be from the AI
            messages = result["messages"]
            last_message = messages[-1]
            
            print(f"\nSarah (Bot): {last_message.content}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    asyncio.run(main())
