from pydantic import BaseModel, Field

class AgentResponseSchema(BaseModel):
    """
    Schema for the final response from the agent.
    MUST be used to format the final output to the user.
    """
    answer: str = Field(..., description="The exact text to be spoken and displayed to the user.")
