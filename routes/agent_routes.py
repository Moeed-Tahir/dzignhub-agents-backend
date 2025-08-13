from fastapi import APIRouter
from agents.brand_designer import get_brand_designer_agent
from agents.seo_specialist import get_seo_agent
from agents.content_creator import get_content_creator_agent
from pydantic import BaseModel

router = APIRouter()
# Define request model
class PromptRequest(BaseModel):
    prompt: str


@router.post("/brand-designer")
def brand_designer_endpoint(request: PromptRequest):
    agent = get_brand_designer_agent()
    response = agent.handle_query(request.prompt)  # Changed from .run() to .handle_query()
    return {"response": response}


# Add a GET endpoint for testing
@router.get("/brand-designer")
def brand_designer_test():
    return {"message": "Brand Designer Agent is running! Send a POST request with 'prompt' to interact."}