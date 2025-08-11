from fastapi import FastAPI
from routes import agent_routes

app = FastAPI()

app.include_router(agent_routes.router, prefix="/agents", tags=["Agents"])

@app.get("/")
def root():
    return {"message": "Multi-Agent Backend Running 🚀"}
