from pymongo import MongoClient
from datetime import datetime
from bson import ObjectId
from core.config import MONGODB_URL, DATABASE_NAME
# MongoDB connection
client = MongoClient(MONGODB_URL)
db = client[DATABASE_NAME]

# Collections
users_collection = db.users 
conversations_collection = db.conversations
messages_collection = db.messages

class MongoDB:
    @staticmethod
    def create_conversation(user_id: str, agent: str, title: str = None):
        """Create a new conversation"""
        conversation = {
            "userId": ObjectId(user_id),
            "agent": agent,
            "title": title or f"Chat with {agent}",
            "createdAt": datetime.utcnow(),
            "updatedAt": datetime.utcnow()
        }
        result = conversations_collection.insert_one(conversation)
        return str(result.inserted_id)

    @staticmethod
    def save_message(conversation_id: str, user_id: str, sender: str, text: str, agent: str = None):
        """Save a message to MongoDB"""
        message = {
            "conversationId": ObjectId(conversation_id),
            "userId": ObjectId(user_id),
            "sender": sender,  # 'user' or 'agent'
            "text": text,
            "createdAt": datetime.utcnow(),
            "updatedAt": datetime.utcnow()
        }
        if agent and sender == 'agent':
            message["agent"] = agent
            
        result = messages_collection.insert_one(message)
        return str(result.inserted_id)

    @staticmethod
    def get_conversation_messages(conversation_id: str, user_id: str = None):
        """Get all messages for a conversation"""
        query = {"conversationId": ObjectId(conversation_id)}
        if user_id:
            # Verify the conversation belongs to the user
            conversation = conversations_collection.find_one({
                "_id": ObjectId(conversation_id),
                "userId": ObjectId(user_id)
            })
            if not conversation:
                return []
            
        messages = list(messages_collection.find(query).sort("createdAt", 1))
        
        # Convert ObjectId to string for JSON serialization
        for message in messages:
            message["_id"] = str(message["_id"])
            message["conversationId"] = str(message["conversationId"])
            message["userId"] = str(message["userId"])
            
        return messages
    
    @staticmethod
    def get_conversation_by_id(conversation_id: str):
        """Get a specific conversation by ID"""
        try:
            # Convert string to ObjectId if needed
            if isinstance(conversation_id, str):
                conversation_id = ObjectId(conversation_id)
            
            conversation = conversations_collection.find_one({"_id": conversation_id})
            
            if conversation:
                conversation["_id"] = str(conversation["_id"])
                conversation["userId"] = str(conversation["userId"])  # Convert ObjectId to string
                return conversation
            return None
            
        except Exception as e:
            print(f"Error getting conversation by ID: {e}")
            return None



    @staticmethod
    def get_user_conversations(user_id: str):
        """Get all conversations for a user"""
        conversations = list(conversations_collection.find({"userId": ObjectId(user_id)}).sort("updatedAt", -1))
        
        # Convert ObjectId to string
        for conv in conversations:
            conv["_id"] = str(conv["_id"])
            conv["userId"] = str(conv["userId"])
            
        return conversations
    
    @staticmethod
    def get_user_conversations_by_agent(user_id: str, agent: str):
        """Get all conversations for a user filtered by agent"""
        conversations = list(conversations_collection.find({"userId": ObjectId(user_id), "agent": agent}).sort("updatedAt", -1))
        
        # Convert ObjectId to string
        for conv in conversations:
            conv["_id"] = str(conv["_id"])
            conv["userId"] = str(conv["userId"])
            
        return conversations

    @staticmethod
    def get_user_recent_messages(user_id: str, limit: int = 50):
        """Get recent messages for a user across all conversations"""
        messages = list(messages_collection.find({"userId": ObjectId(user_id)}).sort("createdAt", -1).limit(limit))
        
        # Convert ObjectId to string
        for message in messages:
            message["_id"] = str(message["_id"])
            message["conversationId"] = str(message["conversationId"])
            message["userId"] = str(message["userId"])
            
        return messages

    @staticmethod
    def update_conversation_title(conversation_id: str, title: str):
        """Update conversation title"""
        result = conversations_collection.update_one(
            {"_id": ObjectId(conversation_id)},
            {"$set": {"title": title, "updatedAt": datetime.utcnow()}}
        )
        return result.modified_count > 0

    @staticmethod
    def delete_conversation(conversation_id: str, user_id: str):
        """Delete a conversation and all its messages"""
        # First verify the conversation belongs to the user
        conversation = conversations_collection.find_one({
            "_id": ObjectId(conversation_id),
            "userId": ObjectId(user_id)
        })
        if not conversation:
            raise Exception("Conversation not found or access denied")

        # Delete all messages in the conversation
        messages_result = messages_collection.delete_many({
            "conversationId": ObjectId(conversation_id)
        })

        # Delete the conversation
        conversations_collection.delete_one({
            "_id": ObjectId(conversation_id),
            "userId": ObjectId(user_id)
        })

        return messages_result.deleted_count

    @staticmethod
    def delete_message(message_id: str, user_id: str):
        """Delete a specific message"""
        # Verify the message belongs to the user
        result = messages_collection.delete_one({
            "_id": ObjectId(message_id),
            "userId": ObjectId(user_id)
        })
        
        if result.deleted_count == 0:
            raise Exception("Message not found or access denied")
        
        return True

    @staticmethod
    def get_conversation_by_id(conversation_id: str, user_id: str = None):
        """Get a specific conversation"""
        query = {"_id": ObjectId(conversation_id)}
        if user_id:
            query["userId"] = ObjectId(user_id)
            
        conversation = conversations_collection.find_one(query)
        if conversation:
            conversation["_id"] = str(conversation["_id"])
            conversation["userId"] = str(conversation["userId"])
            
        return conversation

    @staticmethod
    def get_user_brand_design(user_id: str):
        """Get user's brandDesign field"""
        try:
            user = users_collection.find_one(
                {"_id": ObjectId(user_id)},
                {"brandDesign": 1}
            )
            return user.get("brandDesign", {}) if user else {}
        except Exception as e:
            print(f"[DEBUG] Error getting brand design: {e}")
            return {}

    @staticmethod
    def update_user_brand_design(user_id: str, brand_design_data: dict):
        """Update user's brandDesign field"""
        try:
            result = users_collection.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": {"brandDesign": brand_design_data}}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"[DEBUG] Error updating brand design: {e}")
            return False