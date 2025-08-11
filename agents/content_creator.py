from core.base_agent import BaseAgent
from core.config import ANTHROPIC_API_KEY
from langchain_anthropic import ChatAnthropic

def get_content_creator_agent():
    llm = ChatAnthropic(
        model="claude-3.5-sonnet",
        temperature=0.7,
        api_key=ANTHROPIC_API_KEY
    )
    return BaseAgent(llm)
