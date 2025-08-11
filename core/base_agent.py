from langchain.schema import HumanMessage
from typing import Any

class BaseAgent:
    def __init__(self, llm):
        self.llm = llm

    def run(self, prompt: str) -> Any:
        response = self.llm([HumanMessage(content=prompt)])
        return response.content if hasattr(response, "content") else str(response)
