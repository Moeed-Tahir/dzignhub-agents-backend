from core.base_agent import BaseAgent
from core.config import GOOGLE_API_KEY
from langchain_google_genai import ChatGoogleGenerativeAI

def get_seo_agent():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest",
        temperature=0.5,
        google_api_key=GOOGLE_API_KEY
    )
    return BaseAgent(llm)
