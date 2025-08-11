# agents/brand_designer.py
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
import base64
import os
from core.config import OPENAI_API_KEY

# DALL·E tool for image generation
def generate_logo_dalle(info: dict):
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = f"Design a {info['logo_type']} logo for {info['brand_name']}, targeting {info['target_audience']}. Use these colors: {info['color_palette']}."
    result = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="1024x1024"
    )

    image_url = result.data[0].url
    return f"Logo generated successfully! View it here: {image_url}"

# Brand Designer Agent Class
class BrandDesignerAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4.1", temperature=0.7)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Data store for collected info
        self.design_info = {
            "brand_name": None,
            "logo_type": None,
            "target_audience": None,
            "color_palette": None
        }

        tools = [
            Tool(
                name="Generate Logo with DALL·E",
                func=lambda _: generate_logo_dalle(self.design_info),
                description="Generate a logo once all brand details are collected"
            )
        ]

        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True
        )

    def handle_query(self, query: str):
        # Step 1: Check if user wants a logo
        if "logo" in query.lower():
            missing = [k for k, v in self.design_info.items() if not v]

            # Ask for missing details
            if missing:
                next_missing = missing[0]
                questions = {
                    "brand_name": "What's your brand name?",
                    "logo_type": "What type of logo do you want? (text, icon, mascot, etc.)",
                    "target_audience": "Who is your target audience?",
                    "color_palette": "Any specific color palette?"
                }
                return questions[next_missing]
            else:
                # All info available — trigger logo generation
                return generate_logo_dalle(self.design_info)

        # Step 2: Store answers if they are responses to previous questions
        for key in self.design_info.keys():
            if not self.design_info[key] and query.strip():
                self.design_info[key] = query
                return "Got it! " + self.handle_query("logo")

        # Step 3: Fallback to normal conversation
        return self.agent.run(query)

def get_brand_designer_agent():
    """Factory function to create and return a BrandDesignerAgent instance"""
    return BrandDesignerAgent()
