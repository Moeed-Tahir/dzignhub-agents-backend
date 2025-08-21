import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API")
GOOGLE_API_KEY = os.getenv("GEMINI_API")
ANTHROPIC_API = os.getenv("ANTHROPIC_API")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")


# MongoDB Configuration
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb+srv://abdullahk10204:fB3RFeBuhMtz5RXj@cluster0.i7klkt2.mongodb.net/")
DATABASE_NAME = os.getenv("DATABASE_NAME", "allmyai")