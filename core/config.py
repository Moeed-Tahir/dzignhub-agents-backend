import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API")
GOOGLE_API_KEY = os.getenv("GEMINI_API")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API")
