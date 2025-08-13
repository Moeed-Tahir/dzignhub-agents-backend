# agents/brand_designer.py
import os
from openai import OpenAI

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool, AgentType
from core.config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENV
from pinecone import Pinecone, ServerlessSpec

# ---------------------------
# Pinecone Setup
# ---------------------------
pinecone = Pinecone(
        api_key=PINECONE_API_KEY
    )
INDEX_NAME = "ai-agents-memory"
# Check if index exists using the new API
existing_indexes = [index.name for index in pinecone.list_indexes()]

if INDEX_NAME not in existing_indexes:
    pinecone.create_index(
        name=INDEX_NAME,
        dimension=1536,  # For text-embedding-3-small
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",  # or "gcp", "azure"
            region="us-east-1"  # Choose your preferred region
        )
    )

pinecone_index = pinecone.Index(INDEX_NAME)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------
# Pinecone Helper Functions
# ---------------------------
def embed_text(text: str):
    """Create an embedding for given text."""
    return openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

def store_in_pinecone(agent_type: str, role: str, text: str):
    """Store only embedding in Pinecone (no raw text)."""
    vector = embed_text(text)
    vector_id = f"{agent_type}-{role}-{hash(text)}"
    pinecone_index.upsert([(vector_id, vector)])

def retrieve_from_pinecone(query: str, top_k: int = 3):
    """Retrieve most relevant embeddings for a query."""
    query_vector = embed_text(query)
    results = pinecone_index.query(vector=query_vector, top_k=top_k, include_values=False)
    return results

# ---------------------------
# DALL·E Tool
# ---------------------------
def generate_logo_dalle(info: dict):
    prompt = f"Design a {info['logo_type']} logo for {info['brand_name']}, targeting {info['target_audience']}. Use these colors: {info['color_palette']}."
    result = openai_client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="1024x1024"
    )
    return f"Logo generated successfully! View it here: {result.data[0].url}"

# ---------------------------
# Brand Designer Agent
# ---------------------------
class BrandDesignerAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4.1", temperature=0.7)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
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
        # Store user query in Pinecone
        store_in_pinecone("brand-designer", "user", query)

        # Retrieve similar past queries (for potential context)
        past_results = retrieve_from_pinecone(query)
        if past_results.matches:
            print(f"[DEBUG] Similar past entries found: {past_results.matches}")

        # Step 1: Check if user wants a logo
        if "logo" in query.lower():
            missing = [k for k, v in self.design_info.items() if not v]
            if missing:
                next_missing = missing[0]
                questions = {
                    "brand_name": "What's your brand name?",
                    "logo_type": "What type of logo do you want? (text, icon, mascot, etc.)",
                    "target_audience": "Who is your target audience?",
                    "color_palette": "Any specific color palette?"
                }
                ai_response = questions[next_missing]
                store_in_pinecone("brand-designer", "assistant", ai_response)
                return ai_response
            else:
                ai_response = generate_logo_dalle(self.design_info)
                store_in_pinecone("brand-designer", "assistant", ai_response)
                return ai_response

        # Step 2: Store missing design info
        for key in self.design_info.keys():
            if not self.design_info[key] and query.strip():
                self.design_info[key] = query
                return "Got it! " + self.handle_query("logo")

        # Step 3: Fallback to normal conversation
        ai_response = self.agent.run(query)
        store_in_pinecone("brand-designer", "assistant", ai_response)
        return ai_response

def get_brand_designer_agent():
    return BrandDesignerAgent()
