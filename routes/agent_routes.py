from fastapi import APIRouter
from agents.brand_designer import get_brand_designer_agent
from agents.seo_specialist import get_seo_agent
from agents.content_creator import get_content_creator_agent

router = APIRouter()

@router.post("/brand-designer")
def brand_designer_endpoint(prompt: str):
    agent = get_brand_designer_agent()
    return {"response": agent.run(prompt)}
