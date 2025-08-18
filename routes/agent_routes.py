from fastapi import APIRouter, Query, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from agents.brand_designer import get_brand_designer_agent
from core.database import MongoDB
from agents.brand_designer import search_conversations_by_query

router = APIRouter()

# Request Models
class ChatRequest(BaseModel):
    prompt: str
    user_id: str
    conversation_id: Optional[str] = None

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

# Agent Chat Routes
@router.post("/brand-designer")
def brand_designer_endpoint(request: ChatRequest):
    """Brand designer endpoint with automatic context handling"""
    try:
        # Create conversation if not provided
        conversation_id = request.conversation_id
        is_new_conversation = False
        
        if not conversation_id:
            conversation_id = MongoDB.create_conversation(
                user_id=request.user_id,
                agent="brand-designer",
                title="Brand Design Chat"
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