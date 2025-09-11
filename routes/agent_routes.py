from fastapi import APIRouter, Query, HTTPException, Depends
from pydantic import BaseModel, validator
from typing import Optional
from agents.brand_designer import get_brand_designer_agent
from core.database import MongoDB
from agents.brand_designer import search_conversations_by_query
from agents.content_creator import get_content_creator_agent, search_content_conversations
from agents.seo_specialist import get_seo_specialist_agent, search_seo_conversations
from agents.mira_strategist import get_mira_strategist_agent, search_strategy_conversations
from openai import OpenAI
import os
from fastapi.responses import StreamingResponse
import json
import asyncio
from typing import AsyncGenerator
from groq import Groq
router = APIRouter()
from core.config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENV, GROQ_API_KEY, SEARCHAPI_KEY
from agents.pitch_deck import get_pitch_deck_agent, search_conversations_by_query as search_pitch_deck_conversations
from agents.super_agent import get_super_agent, search_super_agent_conversations

# Request Models
class ChatRequest(BaseModel):
    prompt: str
    user_id: str
    conversation_id: Optional[str] = None

    @validator('conversation_id', pre=True)
    def normalize_conversation_id(cls, v):
        if not v or v in ['', 'null', 'undefined', 'None']:
            return None
        return v

class ChatRequestPitchDeck(BaseModel):
    prompt: str
    user_id: str
    conversation_id: Optional[str] = None
    selectedTemplate: Optional[str] = None 

    @validator('conversation_id', pre=True)
    def normalize_conversation_id(cls, v):
        if not v or v in ['', 'null', 'undefined', 'None']:
            return None
        return v

class NewConversationRequest(BaseModel):
    user_id: str
    agent: str
    title: Optional[str] = None

class MessageRequest(BaseModel):
    conversation_id: str
    user_id: str
    sender: str  # 'user' or 'agent'
    text: str
    agent: Optional[str] = None

class UpdateConversationRequest(BaseModel):
    title: str

# Conversation Routes
@router.post("/conversations")
def create_conversation(request: NewConversationRequest):
    """Create a new conversation"""
    try:
        conversation_id = MongoDB.create_conversation(
            user_id=request.user_id,
            agent=request.agent,
            title=request.title
        )
        return {
            "success": True,
            "conversation_id": conversation_id,
            "message": "Conversation created successfully"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/generate-immediate-response")
async def generate_immediate_response(request: dict):
    """Generate a dynamic, conversational immediate response using Groq"""
    try:
        user_input = request.get("user_input", "")
        if not user_input:
            raise HTTPException(status_code=400, detail="User input is required")
        
        # Initialize Groq client
        groq_client = Groq(api_key=GROQ_API_KEY)
        
        # Updated prompt to generate responses like "I will generate [this]" based on user input
        prompt = f"Generate a brief, engaging, and conversational response (under 50 words) starting with 'I will generate' followed by what the user is requesting, based on this user input for an AI design assistant: '{user_input}'. Make it friendly."
        
        # Use the specified model
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Updated model as specified
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,  # Keep short for speed
            temperature=0.7,  # Balanced creativity
            timeout=3  # 3-second timeout to match your limit
        )
        
        ai_response = response.choices[0].message.content.strip()
        
        return {
            "success": True,
            "response": ai_response,
            "model": "llama-3.1-8b-instant"
        }
    except Exception as e:
        print(f"[DEBUG] Groq immediate response failed: {e}")
        # Fallback to static response
        fallback_response = generate_immediate_response_fallback(user_input)
        return {
            "success": False,
            "response": fallback_response,
            "error": str(e)
        }

def generate_immediate_response_fallback(user_input: str) -> str:
    input_lower = user_input.lower()
    if "logo" in input_lower:
        return "🎨 I'll create a professional logo for you! Let's start by analyzing your requirements..."
    elif "instagram" in input_lower and ("post" in input_lower or "poster" in input_lower):
        return "📱 I'll design an Instagram post for you! Let's gather the details..."
    # ...add more fallbacks as needed...
    else:
        return "💭 I'm analyzing your request and will help you create what you need! Let's get started..."


@router.get("/conversations/search")
def search_conversations(
    query: str = Query(..., description="Search query"),
    user_id: str = Query(..., description="User ID"),
    agent: str = Query("brand-designer", description="Agent type"),
    limit: int = Query(10, description="Number of results to return")
):
    """Search conversations using vector similarity"""
    try:
        # Search for similar conversations
        search_results = search_conversations_by_query(
            query=query,
            user_id=user_id,
            agent_type=agent,
            top_k=limit
        )
        
        return {
            "success": True,
            "results": search_results,
            "count": len(search_results),
            "query": query
        }
    except Exception as e:
        print(f"[DEBUG] Search route error: {e}")
        return {"success": False, "error": str(e)}
    

@router.post("/save-rich-message")
async def save_rich_message(request: dict):
    """Save a complete rich message object"""
    try:
        conversation_id = request.get('conversation_id')
        message_data = request.get('message', {})
        user_id = request.get('user_id')
        
        if not conversation_id or not message_data:
            raise HTTPException(status_code=400, detail="Missing conversation_id or message data")
        
        # Get the brand designer agent and save
        agent = get_brand_designer_agent(user_id, conversation_id)
        result = agent.save_rich_message(conversation_id, user_id, message_data)
        
        if result["type"] == "success":
            return result
        else:
            raise HTTPException(status_code=500, detail=result["message"])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save message: {str(e)}")



@router.get("/conversations/single-agent/{agent}/{userId}")
def get_user_conversations_by_agent(
    agent: str, 
    userId: str
):
    """Get all conversations for a user filtered by agent (JWT protected)"""
    try:
        # Now using the agent parameter correctly
        conversations = MongoDB.get_user_conversations_by_agent(userId, agent)
        return {
            "success": True, 
            "conversations": conversations,
            "count": len(conversations),
            "user_id": userId,  # Include for debugging
            "agent": agent
        }
    except Exception as e:
        return {"success": False, "error": str(e)}



@router.get("/conversations/{user_id}")
def get_user_conversations(user_id: str):
    """Get all conversations for a user"""
    try:
        conversations = MongoDB.get_user_conversations(user_id)
        return {
            "success": True, 
            "conversations": conversations,
            "count": len(conversations)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.put("/conversations/{conversation_id}")
def update_conversation_title(conversation_id: str, request: UpdateConversationRequest):
    """Update conversation title"""
    try:
        MongoDB.update_conversation_title(conversation_id, request.title)
        return {
            "success": True,
            "message": "Conversation title updated successfully"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# Message Routes
@router.post("/messages")
def save_message(request: MessageRequest):
    """Save a message to MongoDB"""
    try:
        message_id = MongoDB.save_message(
            conversation_id=request.conversation_id,
            user_id=request.user_id,
            sender=request.sender,
            text=request.text,
            agent=request.agent
        )
        return {
            "success": True,
            "message_id": message_id,
            "message": "Message saved successfully"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.get("/conversations/{conversation_id}/messages")
def get_conversation_messages(conversation_id: str, user_id: str = Query(...)):
    """Get all messages in a conversation"""
    try:
        messages = MongoDB.get_conversation_messages(conversation_id, user_id)
        return {
            "success": True, 
            "messages": messages,
            "count": len(messages)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.get("/messages/user/{user_id}")
def get_user_messages(user_id: str, limit: int = Query(50, description="Number of messages to return")):
    """Get recent messages for a user across all conversations"""
    try:
        # This would need to be added to database.py
        messages = MongoDB.get_user_recent_messages(user_id, limit)
        return {
            "success": True,
            "messages": messages,
            "count": len(messages)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# Function to generate conversation title based on prompt | conditional
def generate_conversation_title(prompt: str) -> str:
    """Generate a conversation title based on the user's first message"""
    try:
        # Clean and truncate the prompt
        prompt_clean = prompt.strip().lower()
        
        # Define keywords and their corresponding titles
        title_patterns = {
            # Logo-related
            "logo": "Logo Design",
            "brand logo": "Brand Logo Design", 
            "company logo": "Company Logo Design",
            "logo design": "Logo Design Project",
            
            # Social media
            "instagram": "Instagram Design",
            "linkedin": "LinkedIn Design", 
            "facebook": "Facebook Design",
            "social media": "Social Media Design",
            "post design": "Social Media Post",
            
            # Brand identity
            "brand": "Brand Identity",
            "branding": "Brand Strategy",
            "brand identity": "Brand Identity Design",
            "visual identity": "Visual Identity",
            
            # Colors
            "color": "Brand Colors", 
            "color scheme": "Color Scheme Design",
            "palette": "Color Palette",
            
            # Business cards & print
            "business card": "Business Card Design",
            "card design": "Business Card",
            "flyer": "Flyer Design",
            "brochure": "Brochure Design",
            
            # Web design
            "website": "Website Design",
            "web design": "Web Design Project",
            "landing page": "Landing Page Design",
            
            # Specific design types
            "poster": "Poster Design",
            "banner": "Banner Design", 
            "cover": "Cover Design",
            "header": "Header Design"
        }
        
        # Check for specific patterns (most specific first)
        for keyword, title in title_patterns.items():
            if keyword in prompt_clean:
                return title
        
        # If no specific pattern found, create title from first few words
        words = prompt.split()
        if len(words) <= 3:
            return prompt.title()
        else:
            # Take first 3-4 words and add "Design"
            title_words = words[:3]
            title = " ".join(title_words).title()
            
            # Add "Design" if not already present
            if "design" not in title.lower():
                title += " Design"
                
            return title
            
    except Exception as e:
        print(f"[DEBUG] Title generation error: {e}")
        return "Brand Design Chat"
    
# Function to generate AI-powered title using OpenAI (fallback to keyword-based)
def generate_dynamic_title(prompt: str) -> str:
    """Generate title using OpenAI (fallback to keyword-based)"""
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Generate a short, descriptive title (3-5 words) for a conversation based on the user's request. Focus on the main  task."},
                {"role": "user", "content": f"User request: {prompt}"}
            ],
            max_tokens=20,
            temperature=0.3
        )
        
        ai_title = response.choices[0].message.content.strip()
        
        # Clean up the title (remove quotes, etc.)
        ai_title = ai_title.replace('"', '').replace("'", '')
        
        # Validate length
        if len(ai_title) > 50:
            raise Exception("Title too long")
            
        return ai_title
        
    except Exception as e:
        print(f"[DEBUG] AI title generation failed: {e}, using keyword-based")
        return generate_conversation_title(prompt)  # Fallback to keyword-based

@router.post("/brand-designer")
def brand_designer_endpoint(request: ChatRequest):
    """Brand designer endpoint with automatic context handling"""
    try:
        # Create conversation if not provided
        conversation_id = request.conversation_id
        is_new_conversation = False
        
        if not conversation_id:
            # Use AI-powered title generation with keyword fallback
            dynamic_title = generate_dynamic_title(request.prompt)
            
            conversation_id = MongoDB.create_conversation(
                user_id=request.user_id,
                agent="brand-designer", 
                title=dynamic_title
            )
            is_new_conversation = True
            

        # Get agent and handle query (context is automatically handled inside)
        agent = get_brand_designer_agent(
            user_id=request.user_id,
            conversation_id=conversation_id
        )
        
        # Single method call - context detection is automatic
        response = agent.handle_query(request.prompt)
        
        return {
            "success": True,
            "response": response,
            "conversation_id": conversation_id,
            "is_new_conversation": is_new_conversation,
            "agent": "brand-designer"
        }
    except Exception as e:
        print(f"Error in brand designer endpoint: {e}")
        return {"success": False, "error": str(e)}
    

# Utility Routes
@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, user_id: str = Query(...)):
    """Delete a conversation and all its messages"""
    try:
        # This would need to be added to database.py
        deleted_count = MongoDB.delete_conversation(conversation_id, user_id)
        return {
            "success": True,
            "message": f"Conversation and {deleted_count} messages deleted successfully"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.delete("/messages/{message_id}")
def delete_message(message_id: str, user_id: str = Query(...)):
    """Delete a specific message"""
    try:
        # This would need to be added to database.py
        MongoDB.delete_message(message_id, user_id)
        return {
            "success": True,
            "message": "Message deleted successfully"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/brand-designer/stream")
async def brand_designer_stream_endpoint(request: ChatRequest):
    """Brand designer endpoint with streaming responses"""
    try:
        # Create conversation if not provided
        conversation_id = request.conversation_id
        is_new_conversation = False
        print(conversation_id)
        print(request.user_id)
        if conversation_id is None:
            dynamic_title = generate_dynamic_title(request.prompt)
            conversation_id = MongoDB.create_conversation(
                user_id=request.user_id,
                agent="brand-designer", 
                title=dynamic_title
            )
            print("New conversation id: ", conversation_id)
            is_new_conversation = True

        if not conversation_id:
            raise Exception("Failed to create or retrieve conversation ID")


        # Get agent
        agent = get_brand_designer_agent(
            user_id=request.user_id,
            conversation_id=str(conversation_id)
        )
        
        # Create streaming generator
        async def generate_stream():
            try:
                # Send initial response
                initial_data = {
                    "type": "conversation_info",
                    "conversation_id": conversation_id,
                    "is_new_conversation": is_new_conversation,
                    "agent": "brand-designer"
                }
                yield f"data: {json.dumps(initial_data)}\n\n"
                
                # Stream the agent response
                async for chunk in agent.stream_agent_response(request.prompt):
                    print(f"[DEBUG] Streaming chunk: {chunk.get('type', 'unknown')}")
                    yield f"data: {json.dumps(chunk)}\n\n"
                
                # Send completion signal
                completion_data = {"type": "complete"}
                yield f"data: {json.dumps(completion_data)}\n\n"
                
            except Exception as e:
                error_data = {"type": "error", "message": str(e)}
                yield f"data: {json.dumps(error_data)}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )
        
    except Exception as e:
        print(f"Error in brand designer stream endpoint: {e}")
        return {"success": False, "error": str(e)}

# Status Routes
@router.get("/brand-designer")
def brand_designer_status():
    """Get brand designer agent status"""
    return {
        "success": True,
        "message": "Brand Designer Agent is running!",
        "agent": "brand-designer",
        "version": "1.0.0"
    }


@router.post("/content-creator")
def content_creator_endpoint(request: ChatRequest):
    """Content creator endpoint with automatic context handling"""
    try:
        # Create conversation if not provided
        conversation_id = request.conversation_id
        is_new_conversation = False
        
        if not conversation_id:
            # Use AI-powered title generation with keyword fallback
            dynamic_title = generate_dynamic_content_title(request.prompt)
            
            conversation_id = MongoDB.create_conversation(
                user_id=request.user_id,
                agent="content-creator", 
                title=dynamic_title
            )
            is_new_conversation = True

        # Get agent and handle query
        agent = get_content_creator_agent(
            user_id=request.user_id,
            conversation_id=conversation_id
        )
        
        response = agent.handle_query(request.prompt)
        
        return {
            "success": True,
            "response": response,
            "conversation_id": conversation_id,
            "is_new_conversation": is_new_conversation,
            "agent": "content-creator"
        }
    except Exception as e:
        print(f"Error in content creator endpoint: {e}")
        return {"success": False, "error": str(e)}

@router.get("/content-creator")
def content_creator_status():
    """Get content creator agent status"""
    return {
        "success": True,
        "message": "Content Creator Agent is running!",
        "agent": "content-creator",
        "version": "1.0.0"
    }

# content-specific search endpoint
@router.get("/content/search")
def search_content_conversations(
    query: str = Query(..., description="Search query"),
    user_id: str = Query(..., description="User ID"),
    limit: int = Query(10, description="Number of results to return")
):
    """Search content conversations using vector similarity"""
    try:
        search_results = search_content_conversations(
            query=query,
            user_id=user_id,
            agent_type="content-creator",
            top_k=limit
        )
        
        return {
            "success": True,
            "results": search_results,
            "count": len(search_results),
            "query": query
        }
    except Exception as e:
        print(f"[DEBUG] Content search route error: {e}")
        return {"success": False, "error": str(e)}


# Add this endpoint
@router.post("/seo-specialist")
def seo_specialist_endpoint(request: ChatRequest):
    """SEO specialist endpoint with automatic context handling"""
    try:
        # Create conversation if not provided
        conversation_id = request.conversation_id
        is_new_conversation = False
        
        if not conversation_id:
            # Use AI-powered title generation with keyword fallback
            dynamic_title = generate_dynamic_seo_title(request.prompt)
            
            conversation_id = MongoDB.create_conversation(
                user_id=request.user_id,
                agent="seo-specialist", 
                title=dynamic_title
            )
            is_new_conversation = True

        # Get agent and handle query
        agent = get_seo_specialist_agent(
            user_id=request.user_id,
            conversation_id=conversation_id
        )
        
        response = agent.handle_query(request.prompt)
        
        return {
            "success": True,
            "response": response,
            "conversation_id": conversation_id,
            "is_new_conversation": is_new_conversation,
            "agent": "seo-specialist"
        }
    except Exception as e:
        print(f"Error in SEO specialist endpoint: {e}")
        return {"success": False, "error": str(e)}

@router.get("/seo-specialist")
def seo_specialist_status():
    """Get SEO specialist agent status"""
    return {
        "success": True,
        "message": "SEO Specialist Agent is running!",
        "agent": "seo-specialist",
        "version": "1.0.0"
    }

# SEO-specific search endpoint
@router.get("/seo/search")
def search_seo_conversations(
    query: str = Query(..., description="Search query"),
    user_id: str = Query(..., description="User ID"),
    limit: int = Query(10, description="Number of results to return")
):
    """Search SEO conversations using vector similarity"""
    try:
        search_results = search_seo_conversations(
            query=query,
            user_id=user_id,
            agent_type="seo-specialist",
            top_k=limit
        )
        
        return {
            "success": True,
            "results": search_results,
            "count": len(search_results),
            "query": query
        }
    except Exception as e:
        print(f"[DEBUG] SEO search route error: {e}")
        return {"success": False, "error": str(e)}


# Strategy Routes
@router.post("/strategist")
def mira_strategist_endpoint(request: ChatRequest):
    """Mira strategist endpoint with automatic context handling"""
    try:
        # Create conversation if not provided
        conversation_id = request.conversation_id
        is_new_conversation = False
        
        if not conversation_id:
            # Use AI-powered title generation with keyword fallback
            dynamic_title = generate_dynamic_strategy_title(request.prompt)
            
            conversation_id = MongoDB.create_conversation(
                user_id=request.user_id,
                agent="strategist", 
                title=dynamic_title
            )
            is_new_conversation = True

        # Get agent and handle query
        agent = get_mira_strategist_agent(
            user_id=request.user_id,
            conversation_id=conversation_id
        )
        
        response = agent.handle_query(request.prompt)
        
        return {
            "success": True,
            "response": response,
            "conversation_id": conversation_id,
            "is_new_conversation": is_new_conversation,
            "agent": "strategist"
        }
    except Exception as e:
        print(f"Error in Mira strategist endpoint: {e}")
        return {"success": False, "error": str(e)}

@router.get("/strategist")
def mira_strategist_status():
    """Get Mira strategist agent status"""
    return {
        "success": True,
        "message": "Mira Strategist Agent is running!",
        "agent": "strategist",
        "version": "1.0.0"
    }

# Strategy-specific search endpoint
@router.get("/strategy/search")
def search_strategy_conversations(
    query: str = Query(..., description="Search query"),
    user_id: str = Query(..., description="User ID"),
    limit: int = Query(10, description="Number of results to return")
):
    """Search strategy conversations using vector similarity"""
    try:
        search_results = search_strategy_conversations(
            query=query,
            user_id=user_id,
            agent_type="strategist",
            top_k=limit
        )
        
        return {
            "success": True,
            "results": search_results,
            "count": len(search_results),
            "query": query
        }
    except Exception as e:
        print(f"[DEBUG] Strategy search route error: {e}")
        return {"success": False, "error": str(e)}

# Add title generation helper
def generate_dynamic_strategy_title(prompt: str) -> str:
    """Generate title for strategy conversations using OpenAI (fallback to keyword-based)"""
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Generate a short, descriptive title (3-5 words) for a strategic planning conversation based on the user's request. Focus on the strategic objective or planning type."},
                {"role": "user", "content": f"User request: {prompt}"}
            ],
            max_tokens=20,
            temperature=0.3
        )
        
        ai_title = response.choices[0].message.content.strip()
        
        # Clean up the title (remove quotes, etc.)
        ai_title = ai_title.replace('"', '').replace("'", '')
        
        # Validate length
        if len(ai_title) > 50:
            raise Exception("Title too long")
            
        print(f"[DEBUG] Generated AI strategy title: {ai_title}")
        return ai_title
        
    except Exception as e:
        print(f"[DEBUG] AI strategy title generation failed: {e}, using keyword-based")
        return generate_strategy_title_pattern(prompt)  # Fallback to keyword-based

def generate_strategy_title_pattern(prompt: str) -> str:
    """Generate title for strategy conversations using keyword patterns"""
    try:
        prompt_clean = prompt.strip().lower()
        
        title_patterns = {
            # Strategic Planning
            "business strategy": "Business Strategy",
            "strategic plan": "Strategic Planning",
            "growth strategy": "Growth Strategy",
            "business plan": "Business Planning",
            "strategy": "Strategic Planning",
            
            # Goal Setting
            "goal setting": "Goal Planning",
            "goals": "Goal Strategy",
            "objectives": "Objective Planning",
            "targets": "Target Planning",
            
            # Market & Positioning
            "market strategy": "Market Strategy",
            "positioning": "Positioning Strategy",
            "competitive strategy": "Competitive Strategy",
            "market entry": "Market Entry Plan",
            
            # Growth & Scaling
            "growth plan": "Growth Planning",
            "scaling": "Scaling Strategy",
            "expansion": "Expansion Plan",
            "business growth": "Growth Strategy",
            
            # Launch & Development
            "product launch": "Launch Strategy",
            "launch plan": "Launch Planning",
            "startup strategy": "Startup Strategy",
            "mvp strategy": "MVP Planning",
            
            # Revenue & Business Model
            "revenue strategy": "Revenue Planning",
            "business model": "Business Model",
            "monetization": "Monetization Strategy",
            "pricing strategy": "Pricing Strategy",
            
            # Team & Operations
            "team strategy": "Team Planning",
            "operational strategy": "Operations Strategy",
            "resource planning": "Resource Planning",
            
            # General
            "roadmap": "Strategic Roadmap",
            "planning": "Strategic Planning",
            "vision": "Vision Planning",
            "mission": "Mission Planning"
        }
        
        # Check for specific patterns (most specific first)
        for keyword, title in title_patterns.items():
            if keyword in prompt_clean:
                return title
        
        # Extract strategic terms from prompt
        strategy_terms = ['strategy', 'plan', 'goal', 'vision', 'growth', 'business']
        found_terms = []
        
        words = prompt.split()
        for word in words:
            if word.lower() in strategy_terms or any(term in word.lower() for term in strategy_terms):
                found_terms.append(word)
                if len(found_terms) >= 2:
                    break
        
        if found_terms:
            title = " ".join(found_terms).title()
            if "strategy" not in title.lower() and "plan" not in title.lower():
                title += " Strategy"
            return title
        
        # Default patterns based on action words
        if any(word in prompt_clean for word in ['grow', 'scale', 'expand', 'increase']):
            return "Growth Strategy"
        elif any(word in prompt_clean for word in ['launch', 'start', 'begin', 'create']):
            return "Launch Planning"
        elif any(word in prompt_clean for word in ['improve', 'optimize', 'enhance']):
            return "Strategy Optimization"
        
        return "Strategic Planning"
        
    except Exception as e:
        print(f"[DEBUG] Strategy title pattern generation error: {e}")
        return "Strategy Chat"


def generate_dynamic_content_title(prompt: str) -> str:
    """Generate title for content creation conversations using OpenAI (fallback to keyword-based)"""
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Generate a short, descriptive title (3-5 words) for a content creation conversation based on the user's request. Focus on the content type and platform."},
                {"role": "user", "content": f"User request: {prompt}"}
            ],
            max_tokens=20,
            temperature=0.3
        )
        
        ai_title = response.choices[0].message.content.strip()
        
        # Clean up the title (remove quotes, etc.)
        ai_title = ai_title.replace('"', '').replace("'", '')
        
        # Validate length
        if len(ai_title) > 50:
            raise Exception("Title too long")
            
        print(f"[DEBUG] Generated AI content title: {ai_title}")
        return ai_title
        
    except Exception as e:
        print(f"[DEBUG] AI content title generation failed: {e}, using keyword-based")
        return generate_content_title_pattern(prompt)  # Fallback to keyword-based

def generate_content_title_pattern(prompt: str) -> str:
    """Generate title for content creation conversations using keyword patterns"""
    try:
        prompt_clean = prompt.strip().lower()
        
        title_patterns = {
            # Platform-specific content
            "instagram caption": "Instagram Caption",
            "instagram post": "Instagram Content",
            "instagram story": "Instagram Story",
            "instagram": "Instagram Content",
            
            "linkedin post": "LinkedIn Post",
            "linkedin article": "LinkedIn Article", 
            "linkedin": "LinkedIn Content",
            
            "facebook post": "Facebook Post",
            "facebook ad": "Facebook Ad",
            "facebook": "Facebook Content",
            
            "twitter post": "Twitter Post",
            "twitter thread": "Twitter Thread",
            "twitter": "Twitter Content",
            
            "youtube script": "YouTube Script",
            "youtube description": "YouTube Description",
            "youtube": "YouTube Content",
            
            "tiktok script": "TikTok Script",
            "tiktok": "TikTok Content",
            
            # Content types
            "blog article": "Blog Article",
            "blog post": "Blog Writing",
            "blog": "Blog Writing",
            
            "newsletter": "Newsletter Content",
            "email campaign": "Email Campaign",
            "email": "Email Content",
            
            "product description": "Product Description",
            "landing page copy": "Landing Page Copy",
            "website copy": "Website Copy",
            
            "press release": "Press Release",
            "case study": "Case Study",
            
            "script": "Script Writing",
            "ad copy": "Ad Copy",
            "sales copy": "Sales Copy",
            
            # General content
            "content strategy": "Content Strategy",
            "content plan": "Content Planning",
            "content calendar": "Content Calendar",
            "content": "Content Creation",
            "write": "Content Writing",
            "copy": "Copywriting"
        }
        
        # Check for specific patterns (most specific first)
        for keyword, title in title_patterns.items():
            if keyword in prompt_clean:
                return title
        
        # Extract main action/content type from prompt
        words = prompt.split()
        if len(words) <= 3:
            return prompt.title()
        else:
            # Take first 2-3 words that seem content-related
            content_words = []
            for word in words[:4]:
                if word.lower() in ['create', 'write', 'generate', 'make', 'post', 'content', 'copy', 'script']:
                    continue
                content_words.append(word)
                if len(content_words) >= 2:
                    break
            
            if content_words:
                title = " ".join(content_words).title()
                if "content" not in title.lower():
                    title += " Content"
                return title
        
        return "Content Creation"
        
    except Exception as e:
        print(f"[DEBUG] Content title pattern generation error: {e}")
        return "Content Chat"

def generate_dynamic_seo_title(prompt: str) -> str:
    """Generate title for SEO conversations using OpenAI (fallback to keyword-based)"""
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Generate a short, descriptive title (3-5 words) for an SEO consultation conversation based on the user's request. Focus on the SEO service or optimization type."},
                {"role": "user", "content": f"User request: {prompt}"}
            ],
            max_tokens=20,
            temperature=0.3
        )
        
        ai_title = response.choices[0].message.content.strip()
        
        # Clean up the title (remove quotes, etc.)
        ai_title = ai_title.replace('"', '').replace("'", '')
        
        # Validate length
        if len(ai_title) > 50:
            raise Exception("Title too long")
            
        print(f"[DEBUG] Generated AI SEO title: {ai_title}")
        return ai_title
        
    except Exception as e:
        print(f"[DEBUG] AI SEO title generation failed: {e}, using keyword-based")
        return generate_seo_title_pattern(prompt)  # Fallback to keyword-based

def generate_seo_title_pattern(prompt: str) -> str:
    """Generate title for SEO conversations using keyword patterns"""
    try:
        prompt_clean = prompt.strip().lower()
        
        title_patterns = {
            # SEO Services
            "seo audit": "SEO Audit",
            "technical seo": "Technical SEO",
            "seo analysis": "SEO Analysis",
            "seo strategy": "SEO Strategy",
            "seo consultation": "SEO Consultation",
            
            # Keyword Research
            "keyword research": "Keyword Research",
            "keyword analysis": "Keyword Analysis", 
            "keyword strategy": "Keyword Strategy",
            "keyword optimization": "Keyword Optimization",
            "keyword": "Keyword Research",
            
            # Content Optimization
            "content brief": "SEO Content Brief",
            "content optimization": "Content Optimization",
            "seo brief": "SEO Content Brief",
            "blog seo": "Blog SEO",
            "article seo": "Article SEO",
            
            # Meta Optimization
            "meta tags": "Meta Tag Optimization",
            "meta description": "Meta Optimization",
            "title tag": "Title Tag Optimization",
            "meta": "Meta Optimization",
            
            # Page Types
            "landing page seo": "Landing Page SEO",
            "homepage seo": "Homepage SEO", 
            "product page seo": "Product Page SEO",
            "service page seo": "Service Page SEO",
            "category page seo": "Category Page SEO",
            
            # Local SEO
            "local seo": "Local SEO Strategy",
            "google my business": "Google My Business",
            "local optimization": "Local SEO",
            "local search": "Local SEO",
            
            # Technical
            "site speed": "Site Speed Optimization",
            "page speed": "Page Speed SEO",
            "mobile seo": "Mobile SEO",
            "core web vitals": "Core Web Vitals",
            "schema markup": "Schema Markup",
            
            # Goals
            "increase traffic": "SEO Traffic Strategy",
            "improve ranking": "SEO Ranking Strategy", 
            "boost visibility": "SEO Visibility",
            "organic traffic": "Organic Traffic SEO",
            "search ranking": "Search Ranking",
            "serp ranking": "SERP Optimization",
            
            # Content Types
            "blog optimization": "Blog SEO",
            "ecommerce seo": "E-commerce SEO",
            "video seo": "Video SEO",
            "image seo": "Image SEO",
            
            # General
            "seo help": "SEO Assistance",
            "seo tips": "SEO Guidance",
            "seo advice": "SEO Consultation",
            "optimize": "SEO Optimization",
            "ranking": "SEO Ranking",
            "traffic": "SEO Traffic",
            "visibility": "SEO Visibility",
            "seo": "SEO Strategy"
        }
        
        # Check for specific patterns (most specific first)
        for keyword, title in title_patterns.items():
            if keyword in prompt_clean:
                return title
        
        # Extract SEO-related terms from prompt
        seo_terms = ['seo', 'optimize', 'ranking', 'traffic', 'keyword', 'search', 'visibility']
        found_terms = []
        
        words = prompt.split()
        for word in words:
            if word.lower() in seo_terms or any(term in word.lower() for term in seo_terms):
                found_terms.append(word)
                if len(found_terms) >= 2:
                    break
        
        if found_terms:
            title = " ".join(found_terms).title()
            if "seo" not in title.lower():
                title += " SEO"
            return title
        
        # Default patterns based on action words
        if any(word in prompt_clean for word in ['improve', 'increase', 'boost', 'enhance']):
            return "SEO Improvement"
        elif any(word in prompt_clean for word in ['analyze', 'audit', 'check', 'review']):
            return "SEO Analysis"
        elif any(word in prompt_clean for word in ['create', 'build', 'develop', 'plan']):
            return "SEO Strategy"
        
        return "SEO Consultation"
        
    except Exception as e:
        print(f"[DEBUG] SEO title pattern generation error: {e}")
        return "SEO Chat"
    



# Pitch Deck Routes
@router.post("/pitch-deck")
def pitch_deck_endpoint(request: ChatRequestPitchDeck):
    """Pitch deck endpoint with automatic context handling"""
    try:
        # Create conversation if not provided
        conversation_id = request.conversation_id
        is_new_conversation = False
        
        if not conversation_id:
            # Use AI-powered title generation with keyword fallback
            dynamic_title = generate_dynamic_pitch_deck_title(request.prompt)
            
            conversation_id = MongoDB.create_conversation(
                user_id=request.user_id,
                agent="pitch-deck", 
                title=dynamic_title
            )
            is_new_conversation = True

        # Get agent and handle query
        agent = get_pitch_deck_agent(
            user_id=request.user_id,
            conversation_id=conversation_id
        )
        
        response = agent.handle_query(request.prompt)
        
        return {
            "success": True,
            "response": response,
            "conversation_id": conversation_id,
            "is_new_conversation": is_new_conversation,
            "agent": "pitch-deck"
        }
    except Exception as e:
        print(f"Error in pitch deck endpoint: {e}")
        return {"success": False, "error": str(e)}

@router.post("/pitch-deck/stream")
async def pitch_deck_stream_endpoint(request: ChatRequestPitchDeck):
    """Pitch deck endpoint with streaming responses"""
    try:
        # Create conversation if not provided
        conversation_id = request.conversation_id
        selectedTemplate = request.selectedTemplate
        is_new_conversation = False
        
        if conversation_id is None:
            dynamic_title = generate_dynamic_pitch_deck_title(request.prompt)
            conversation_id = MongoDB.create_conversation(
                user_id=request.user_id,
                agent="pitch-deck", 
                title=dynamic_title
            )
            print("New conversation id: ", conversation_id)
            is_new_conversation = True

        if not conversation_id:
            raise Exception("Failed to create or retrieve conversation ID")

        # Get agent
        agent = get_pitch_deck_agent(
            user_id=request.user_id,
            conversation_id=str(conversation_id),
            selectedTemplate=selectedTemplate
        )
        
        # Create streaming generator
        async def generate_stream():
            try:
                # Send initial response
                initial_data = {
                    "type": "conversation_info",
                    "conversation_id": conversation_id,
                    "is_new_conversation": is_new_conversation,
                    "agent": "pitch-deck"
                }
                yield f"data: {json.dumps(initial_data)}\n\n"
                
                # Stream the agent response
                async for chunk in agent.handle_query_stream(request.prompt):
                    print(f"[DEBUG] Streaming chunk: {chunk.get('type', 'unknown')}")
                    yield f"data: {json.dumps(chunk)}\n\n"
                
                # Send completion signal
                completion_data = {"type": "complete"}
                yield f"data: {json.dumps(completion_data)}\n\n"
                
            except Exception as e:
                error_data = {"type": "error", "message": str(e)}
                yield f"data: {json.dumps(error_data)}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )
        
    except Exception as e:
        print(f"Error in pitch deck stream endpoint: {e}")
        return {"success": False, "error": str(e)}

@router.get("/pitch-deck")
def pitch_deck_status():
    """Get pitch deck agent status"""
    return {
        "success": True,
        "message": "Pitch Deck Agent is running!",
        "agent": "pitch-deck",
        "version": "1.0.0"
    }

# Pitch deck-specific search endpoint
@router.get("/pitch-deck/search")
def search_pitch_deck_conversations_endpoint(
    query: str = Query(..., description="Search query"),
    user_id: str = Query(..., description="User ID"),
    limit: int = Query(10, description="Number of results to return")
):
    """Search pitch deck conversations using vector similarity"""
    try:
        search_results = search_pitch_deck_conversations(
            query=query,
            user_id=user_id,
            agent_type="pitch-deck",
            top_k=limit
        )
        
        return {
            "success": True,
            "results": search_results,
            "count": len(search_results),
            "query": query
        }
    except Exception as e:
        print(f"[DEBUG] Pitch deck search route error: {e}")
        return {"success": False, "error": str(e)}

# Add title generation helper function for pitch deck
def generate_dynamic_pitch_deck_title(prompt: str) -> str:
    """Generate title for pitch deck conversations using OpenAI (fallback to keyword-based)"""
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Generate a short, descriptive title (3-5 words) for a pitch deck creation conversation based on the user's request. Focus on the business type or presentation purpose."},
                {"role": "user", "content": f"User request: {prompt}"}
            ],
            max_tokens=20,
            temperature=0.3
        )
        
        ai_title = response.choices[0].message.content.strip()
        
        # Clean up the title (remove quotes, etc.)
        ai_title = ai_title.replace('"', '').replace("'", '')
        
        # Validate length
        if len(ai_title) > 50:
            raise Exception("Title too long")
            
        print(f"[DEBUG] Generated AI pitch deck title: {ai_title}")
        return ai_title
        
    except Exception as e:
        print(f"[DEBUG] AI pitch deck title generation failed: {e}, using keyword-based")
        return generate_pitch_deck_title_pattern(prompt)  # Fallback to keyword-based

def generate_pitch_deck_title_pattern(prompt: str) -> str:
    """Generate title for pitch deck conversations using keyword patterns"""
    try:
        prompt_clean = prompt.strip().lower()
        
        title_patterns = {
            # Presentation Types
            "investor pitch": "Investor Pitch Deck",
            "investor presentation": "Investor Presentation",
            "funding pitch": "Funding Pitch Deck",
            "investor deck": "Investor Deck",
            "pitch deck": "Pitch Deck",
            
            # Client Presentations
            "client pitch": "Client Pitch Deck",
            "client presentation": "Client Presentation",
            "proposal deck": "Proposal Deck",
            "sales deck": "Sales Pitch Deck",
            
            # Product Presentations
            "product launch": "Product Launch Deck",
            "product presentation": "Product Presentation",
            "product demo": "Product Demo Deck",
            
            # Company Presentations
            "company presentation": "Company Presentation",
            "business overview": "Business Overview",
            "company profile": "Company Profile Deck",
            
            # Specific Types
            "startup pitch": "Startup Pitch Deck",
            "seed round": "Seed Round Deck",
            "series a": "Series A Pitch Deck",
            "crowdfunding": "Crowdfunding Deck",
            
            # General
            "slide deck": "Slide Deck",
            "presentation": "Presentation Deck",
            "deck": "Pitch Deck",
            "slides": "Presentation Slides"
        }
        
        # Check for specific patterns (most specific first)
        for keyword, title in title_patterns.items():
            if keyword in prompt_clean:
                return title
        
        # Extract business type and presentation type if available
        business_types = ['startup', 'saas', 'tech', 'healthcare', 'ecommerce', 'app', 'platform', 'service']
        presentation_types = ['investor', 'client', 'product', 'sales', 'funding']
        
        found_business = None
        found_presentation = None
        
        words = prompt.split()
        for word in words:
            word_lower = word.lower()
            if not found_business:
                for btype in business_types:
                    if btype in word_lower:
                        found_business = btype.title()
                        break
            
            if not found_presentation:
                for ptype in presentation_types:
                    if ptype in word_lower:
                        found_presentation = ptype.title()
                        break
        
        if found_business and found_presentation:
            return f"{found_business} {found_presentation} Deck"
        elif found_business:
            return f"{found_business} Pitch Deck"
        elif found_presentation:
            return f"{found_presentation} Pitch Deck"
        
        # Check if it contains company/business name
        if "for" in prompt_clean:
            parts = prompt_clean.split("for")
            if len(parts) > 1 and parts[1].strip():
                potential_name = parts[1].strip().split()[0].title()
                if len(potential_name) > 2:  # Ensure it's a reasonable name length
                    return f"{potential_name} Pitch Deck"
        
        return "Pitch Deck"
        
    except Exception as e:
        print(f"[DEBUG] Pitch deck title pattern generation error: {e}")
        return "Pitch Deck"
    



# Add title generation helper function for super agent
def generate_dynamic_super_agent_title(prompt: str) -> str:
    """Generate title for super agent conversations using OpenAI (fallback to keyword-based)"""
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Generate a short, descriptive title (3-5 words) for a comprehensive content generation conversation based on the user's request. Focus on the main task or content type."},
                {"role": "user", "content": f"User request: {prompt}"}
            ],
            max_tokens=20,
            temperature=0.3
        )
        
        ai_title = response.choices[0].message.content.strip()
        
        # Clean up the title (remove quotes, etc.)
        ai_title = ai_title.replace('"', '').replace("'", '')
        
        # Validate length
        if len(ai_title) > 50:
            raise Exception("Title too long")
            
        print(f"[DEBUG] Generated AI super agent title: {ai_title}")
        return ai_title
        
    except Exception as e:
        print(f"[DEBUG] AI super agent title generation failed: {e}, using keyword-based")
        return generate_super_agent_title_pattern(prompt)  # Fallback to keyword-based

def generate_super_agent_title_pattern(prompt: str) -> str:
    """Generate title for super agent conversations using keyword patterns"""
    try:
        prompt_clean = prompt.strip().lower()
        
        title_patterns = {
            # Content Generation
            "generate content": "Content Generation",
            "create content": "Content Creation",
            "write content": "Content Writing",
            "content creation": "Content Creation",
            "content generation": "Content Generation",
            
            # Design & Assets
            "generate logo": "Logo Generation",
            "create logo": "Logo Creation",
            "design logo": "Logo Design",
            "generate design": "Design Generation",
            "create design": "Design Creation",
            
            # Presentations
            "generate slides": "Slide Generation",
            "create slides": "Slide Creation",
            "pitch deck": "Pitch Deck Creation",
            "presentation": "Presentation Creation",
            
            # Business Content
            "business plan": "Business Plan",
            "marketing strategy": "Marketing Strategy",
            "brand strategy": "Brand Strategy",
            "product description": "Product Description",
            
            # General Generation
            "generate": "Content Generation",
            "create": "Content Creation",
            "make": "Content Creation",
            "build": "Content Building",
            "design": "Design Creation"
        }
        
        # Check for specific patterns (most specific first)
        for keyword, title in title_patterns.items():
            if keyword in prompt_clean:
                return title
        
        # Extract main action from prompt
        words = prompt.split()
        if len(words) <= 3:
            return prompt.title()
        else:
            # Take first 2-3 words that seem generation-related
            generation_words = []
            for word in words[:4]:
                if word.lower() in ['generate', 'create', 'make', 'build', 'design', 'write', 'produce']:
                    continue
                generation_words.append(word)
                if len(generation_words) >= 2:
                    break
            
            if generation_words:
                title = " ".join(generation_words).title()
                if "generation" not in title.lower() and "creation" not in title.lower():
                    title += " Generation"
                return title
        
        return "Content Generation"
        
    except Exception as e:
        print(f"[DEBUG] Super agent title pattern generation error: {e}")
        return "Super Agent Chat"

# Super Agent Routes
@router.post("/super-agent")
def super_agent_endpoint(request: ChatRequest):
    """Super agent endpoint with automatic context handling"""
    try:
        # Create conversation if not provided
        conversation_id = request.conversation_id
        is_new_conversation = False
        
        if not conversation_id:
            # Use AI-powered title generation with keyword fallback
            dynamic_title = generate_dynamic_super_agent_title(request.prompt)
            
            conversation_id = MongoDB.create_conversation(
                user_id=request.user_id,
                agent="super-agent", 
                title=dynamic_title
            )
            is_new_conversation = True

        # Get agent and handle query
        agent = get_super_agent(
            user_id=request.user_id,
            conversation_id=conversation_id
        )
        
        # For super agent, we'll use the streaming response but collect it
        async def collect_response():
            async for chunk in agent.stream_response(request.prompt):
                if chunk.get("type") == "content_response":
                    return chunk.get("message", "")
            return "I apologize, but I encountered an issue generating your content."
        
        # Run the async function
        response = asyncio.run(collect_response())
        
        return {
            "success": True,
            "response": response,
            "conversation_id": conversation_id,
            "is_new_conversation": is_new_conversation,
            "agent": "super-agent"
        }
    except Exception as e:
        print(f"Error in super agent endpoint: {e}")
        return {"success": False, "error": str(e)}

@router.post("/super-agent/stream")
async def super_agent_stream_endpoint(request: ChatRequest):
    """Super agent endpoint with streaming responses"""
    try:
        # Create conversation if not provided
        conversation_id = request.conversation_id
        is_new_conversation = False
        
        if conversation_id is None:
            dynamic_title = generate_dynamic_super_agent_title(request.prompt)
            conversation_id = MongoDB.create_conversation(
                user_id=request.user_id,
                agent="super-agent", 
                title=dynamic_title
            )
            print("New conversation id: ", conversation_id)
            is_new_conversation = True

        if not conversation_id:
            raise Exception("Failed to create or retrieve conversation ID")

        # Get agent
        agent = get_super_agent(
            user_id=request.user_id,
            conversation_id=str(conversation_id)
        )
        
        # Create streaming generator
        async def generate_stream():
            try:
                # Send initial response
                initial_data = {
                    "type": "conversation_info",
                    "conversation_id": conversation_id,
                    "is_new_conversation": is_new_conversation,
                    "agent": "super-agent"
                }
                yield f"data: {json.dumps(initial_data)}\n\n"
                
                # Stream the agent response
                async for chunk in agent.stream_response(request.prompt):
                    print(f"[DEBUG] Streaming chunk: {chunk.get('type', 'unknown')}")
                    yield f"data: {json.dumps(chunk)}\n\n"
                
                # Send completion signal
                completion_data = {"type": "complete"}
                yield f"data: {json.dumps(completion_data)}\n\n"
                
            except Exception as e:
                error_data = {"type": "error", "message": str(e)}
                yield f"data: {json.dumps(error_data)}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )
        
    except Exception as e:
        print(f"Error in super agent stream endpoint: {e}")
        return {"success": False, "error": str(e)}

@router.get("/super-agent")
def super_agent_status():
    """Get super agent status"""
    return {
        "success": True,
        "message": "Super Agent is running!",
        "agent": "super-agent",
        "version": "1.0.0",
        "capabilities": [
            "Content Generation",
            "Web Research",
            "Design Inspiration",
            "Comprehensive Responses",
            "Multi-format Output"
        ]
    }

# Super agent-specific search endpoint
@router.get("/super-agent/search")
def search_super_agent_conversations_endpoint(
    query: str = Query(..., description="Search query"),
    user_id: str = Query(..., description="User ID"),
    limit: int = Query(10, description="Number of results to return")
):
    """Search super agent conversations using vector similarity"""
    try:
        search_results = search_super_agent_conversations(
            query=query,
            user_id=user_id,
            agent_type="super-agent",
            top_k=limit
        )
        
        return {
            "success": True,
            "results": search_results,
            "count": len(search_results),
            "query": query
        }
    except Exception as e:
        print(f"[DEBUG] Super agent search route error: {e}")
        return {"success": False, "error": str(e)}
