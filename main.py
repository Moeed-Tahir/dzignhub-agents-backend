from fastapi import FastAPI
from routes import agent_routes
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI Agents Backend", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

# Include routes
app.include_router(agent_routes.router, prefix="/agents", tags=["Agents"])

@app.get("/")
def root():
    return {"message": "Multi-Agent Backend Running 🚀"}
