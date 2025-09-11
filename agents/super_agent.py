import asyncio
from typing import AsyncGenerator, Dict, Any
from groq import Groq
import re
import json
import requests
import time
from datetime import datetime
from core.config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENV, GROQ_API_KEY, SEARCHAPI_KEY
from core.database import MongoDB
from pinecone import Pinecone, ServerlessSpec

# ---------------------------  
# Pinecone Setup
# ---------------------------
pinecone = Pinecone(api_key=PINECONE_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)
INDEX_NAME = "ai-agents-memory"

existing_indexes = [index.name for index in pinecone.list_indexes()]

if INDEX_NAME not in existing_indexes:
    pinecone.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

pinecone_index = pinecone.Index(INDEX_NAME)

# ---------------------------  
# Helper Functions
# ---------------------------
def embed_text(text: str):
    """Create an embedding for given text."""
    import openai
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    return openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

def store_in_pinecone(agent_type: str, role: str, text: str, user_id: str, conversation_id: str = None):
    """Store embedding with minimal metadata in Pinecone"""
    try:
        # Create embedding
        vector = embed_text(text)
        
        # Create minimal metadata
        metadata = {
            "agent_type": agent_type,
            "role": role,
            "conversation_id": conversation_id,
            "user_id": user_id
        }
        
        # Remove None values
        metadata = {k: v for k, v in metadata.items() if v is not None}
        
        # Create unique vector ID
        vector_id = f"{agent_type}-{role}-{hash(text)}-{datetime.utcnow().timestamp()}"
        
        # Store in Pinecone with minimal metadata
        pinecone_index.upsert([(vector_id, vector, metadata)])
        
        print(f"[DEBUG] Stored in Pinecone: {vector_id} with metadata: {metadata}")
        return vector_id
        
    except Exception as e:
        print(f"[DEBUG] Pinecone storage error: {e}")
        return None

def retrieve_from_pinecone(query: str, top_k: int = 3):
    """Retrieve most relevant embeddings for a query."""
    query_vector = embed_text(query)
    results = pinecone_index.query(vector=query_vector, top_k=top_k, include_values=False)
    return results

def search_super_agent_conversations(query: str, user_id: str, agent_type: str = "super-agent", top_k: int = 10):
        """Search super agent conversations using vector similarity"""
        try:
            # Retrieve relevant embeddings
            results = retrieve_from_pinecone(query, top_k=top_k)
            
            # Filter by user_id and agent_type if metadata is available
            filtered_results = []
            for result in results:
                metadata = result.get('metadata', {})
                if metadata.get('user_id') == user_id and metadata.get('agent_type') == agent_type:
                    filtered_results.append({
                        'id': result.get('id'),
                        'score': result.get('score'),
                        'metadata': metadata
                    })
            
            print(f"[DEBUG] Found {len(filtered_results)} relevant super agent conversations")
            return filtered_results
        
        except Exception as e:
            print(f"[DEBUG] Super agent search error: {e}")
            return []
# ---------------------------  
# Super Agent Class
# ---------------------------
class SuperAgent:
    def __init__(self, user_id: str = None, conversation_id: str = None):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.agent_name = "super-agent"
        self.groq_client = groq_client
        self.reasoning_model = "deepseek-r1-distill-llama-70b"
        self.searchapi_key = SEARCHAPI_KEY
        self.search_base_url = "https://www.searchapi.io/api/v1/search"
        
        
        # Load conversation history if conversation_id exists
        if self.conversation_id and self.user_id:
            self.load_conversation_history()
        
        print(f"[DEBUG] Initialized Super Agent with reasoning model: {self.reasoning_model}")

    def load_conversation_history(self):
        """Load previous messages into memory"""
        print(f"Getting messages of userID {self.user_id} from conversation {self.conversation_id}")
        messages = MongoDB.get_conversation_messages(self.conversation_id, self.user_id)
        print(f"[DEBUG] Loaded {len(messages)} messages from conversation history")

    def search_with_keywords(self, keywords: str):
        """Search web content using SearchAPI.io"""
        params = {
            "engine": "google",
            "q": keywords,
            "api_key": self.searchapi_key,
            "num": 8
        }
        try:
            response = requests.get(self.search_base_url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"[DEBUG] Search API error: {response.status_code}")
                return {"organic_results": []}
        except Exception as e:
            print(f"[DEBUG] Web search error: {e}")
            return {"organic_results": []}
        
    

    def convert_to_keywords(self, query: str) -> str:
        """Convert user query to search keywords using Groq"""
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Convert the user query into a keyword optimized for search engine query. Return single keyword only."},
                    {"role": "user", "content": f"User wants: {query}. Extract design-related search keyword."}
                ],
                temperature=0.1,
                max_tokens=50
            )
            keywords = response.choices[0].message.content.strip()
            print(f"[DEBUG] Generated search keywords: {keywords}")
            return keywords
        except Exception as e:
            print(f"[DEBUG] Keyword generation error: {e}")
            return query  # Fallback to original query

    def search_images(self, query: str, num_results: int = 8):
        """Search for design inspiration images"""
        print(f"[DEBUG] Starting image search for: '{query}' with {num_results} results")
        
        # Search queries for relevant platforms
        design_query = f"site:behance.net {query} OR site:dribbble.com {query}"
        
        print(f"[DEBUG] Design query: {design_query}")
        
        all_images = []
        
        # Search design platforms
        try:
            params = {
                "engine": "google_images",
                "q": design_query,
                "api_key": self.searchapi_key,
                "num": num_results
            }
            
            response = requests.get(self.search_base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                images = data.get("images_results", [])
                for img in images[:num_results]:
                    all_images.append({
                        "title": img.get("title", "Design Inspiration"),
                        "source": img.get("source", "Design Platform"),
                        "link": img.get("link", ""),
                        "thumbnail": img.get("thumbnail", ""),
                        "original": img.get("original", "")
                    })
            else:
                print(f"[DEBUG] Image search error: {response.status_code}")
        except Exception as e:
            print(f"[DEBUG] Image search error: {e}")

        print(f"[DEBUG] Final result: {len(all_images)} design inspiration images collected")
        return all_images

    def format_search_results(self, results):
        """Format web search results for display"""
        articles = []
        for item in results.get("organic_results", []):
            articles.append({
                "title": item.get("title"),
                "link": item.get("link"), 
                "source": item.get("source"),
                "snippet": item.get("snippet")
            })
        return articles[:8]

    async def get_real_model_thinking(self, query: str) -> dict:
        """Get REAL model thinking using Groq reasoning model for content generation"""
        
        thinking_prompt = f"""
        <thinking>
        The user is asking me: "{query}"
        
        Let me think step by step about this content generation request:
        
        1. What exactly is the user asking for? (content, design, presentation, etc.)
        2. What type of content do they need? (business plan, marketing materials, technical documentation, creative content, etc.)
        3. What information will I need to create effective content?
        4. What's my strategy for helping them create compelling content?
        5. How can I make this process smooth and efficient?
        6. What research or inspiration would be most valuable?
        7. How should I structure the final output?
        
        I need to analyze this carefully to provide the best possible service.
        </thinking>
        
        Analyze this user request for content generation. Show your complete reasoning process.
        
        USER REQUEST: "{query}"
        
        Think through this step-by-step and show your reasoning.
        """
        
        try:
            response = self.groq_client.chat.completions.create(
                model=self.reasoning_model,
                messages=[
                    {"role": "system", "content": "You are a professional content creation expert. Think through client requests step-by-step, showing your complete reasoning process in <thinking> tags. Adapt your approach based on the type of content being requested."},
                    {"role": "user", "content": thinking_prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            thinking_text = response.choices[0].message.content.strip()
            
            # Extract thinking content from <thinking> tags
            thinking_match = re.search(r'<thinking>(.*?)</thinking>', thinking_text, re.DOTALL)
            thinking_content = thinking_match.group(1).strip() if thinking_match else thinking_text
            
            # Extract the response after thinking
            response_match = re.search(r'</thinking>\s*(.*)', thinking_text, re.DOTALL)
            reasoning_content = response_match.group(1).strip() if response_match else thinking_text
            
            return {
                "thinking": thinking_content,
                "reasoning": reasoning_content,
                "analysis": "Analyzing the user's content generation request",
                "plan": "Developing strategy to create comprehensive content"
            }
            
        except Exception as e:
            print(f"Real thinking generation error: {e}")
            return {
                "thinking": f"I'm analyzing your request: {query}. Let me think about what type of content you need and what information I'll require to create something valuable and comprehensive.",
                "reasoning": "Processing user request for content generation assistance",
                "analysis": "User wants content creation help",
                "plan": "Gather relevant information and create the requested content"
            }


    async def generate_content_response(self, query: str, search_results: list = None, image_results: list = None) -> str:
        """Generate comprehensive text content using Groq based on user query"""
        
        # Build context from search results
        context_text = ""
        if search_results:
            context_text += "\n\nWeb Search Results:\n"
            for i, result in enumerate(search_results[:5]):
                context_text += f"{i+1}. {result.get('title', '')}: {result.get('snippet', '')[:200]}...\n"
        
        if image_results:
            context_text += "\n\nDesign Inspiration Found:\n"
            for i, img in enumerate(image_results[:5]):
                context_text += f"{i+1}. {img.get('title', '')}, Link: {img.get('link', '')}\n"
        
        # Determine if this is a generation request
        generation_keywords = ["generate", "create", "make", "design", "build", "produce"]
        is_generation = any(keyword in query.lower() for keyword in generation_keywords)
        
        if is_generation:
            # Content generation prompt
            prompt = f"""
            You are a comprehensive content creation assistant. The user wants you to generate content for: "{query}"
            
            Based on the following research and inspiration:{context_text}
            
            Generate comprehensive, detailed content that covers:
            1. Complete information and details
            2. Step-by-step guidance if applicable
            3. Best practices and recommendations
            4. Examples and templates where helpful
            5. Professional, actionable content
            
            Make it thorough but well-organized. Use clear headings and sections.
            Focus on being helpful and providing value. Use images of context if you want to add any image, give final response in MD format.
            """
        else:
            # Regular conversation prompt
            prompt = f"""
            You are a helpful assistant. Answer the user's question: "{query}"
            
            Use this context if relevant:{context_text}
            
            Provide a comprehensive, helpful response. Be conversational and professional.
            """
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a comprehensive content creation assistant. Generate detailed, helpful responses based on user requests."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            print(f"[DEBUG] Generated content response using Groq")
            
            return content
            
        except Exception as e:
            print(f"[DEBUG] Content generation error: {e}")
            return f"I apologize, but I encountered an issue generating the content for your request: '{query}'. Please try again or provide more details."

    async def generate_conversational_response(self, tool_name: str, tool_result: str, context: str) -> str:
        """Generate conversational text after tool completion using Groq with specific details"""
        
        # Build a more specific prompt based on the tool type
        if tool_name == "Pitch Deck Information Extractor":
            # For extraction, mention what was extracted
            specific_prompt = f"""
            You are Max, a friendly pitch deck creation assistant. You just completed information extraction.
            
            TOOL COMPLETED: {tool_name}
            RESULT: {tool_result}
            CONTEXT: {context}
            
            Generate a conversational message (2-3 sentences) that:
            1. Specifically mentions what information you extracted (e.g., "I extracted your business name, target audience, and industry")
            2. Shows enthusiasm about having the key details
            3. Hints at what's coming next (like auto-completing missing info or planning slides)
            4. Uses friendly, professional language
            
            Make it specific to the extraction results, not generic.
            """
            
        elif tool_name == "Smart Auto-Completion":
            # For auto-completion, mention what was completed
            specific_prompt = f"""
            You are Max, a friendly pitch deck creation assistant. You just completed smart auto-completion.
            
            TOOL COMPLETED: {tool_name}
            RESULT: {tool_result}
            CONTEXT: {context}
            
            Generate a conversational message (2-3 sentences) that:
            1. Specifically mentions what fields you auto-completed (e.g., "I auto-completed your market size, business model, and funding ask")
            2. Explains why this helps the pitch deck
            3. Shows enthusiasm about having a complete information set
            4. Uses friendly, professional language
            
            Make it specific to what was completed, not generic.
            """
            
        elif tool_name == "Recent Generation Check":
            # For database check, mention if found or not
            specific_prompt = f"""
            You are Max, a friendly pitch deck creation assistant. You just checked for recent presentations.
            
            TOOL COMPLETED: {tool_name}
            RESULT: {tool_result}
            CONTEXT: {context}
            
            Generate a conversational message (2-3 sentences) that:
            1. Specifically mentions whether you found a recent presentation or not
            2. If found, mention it's ready to use
            3. If not found, express enthusiasm about creating a new one
            4. Uses friendly, professional language
            
            Make it specific to the check results.
            """
            
        else:
            # Generic prompt for other tools
            specific_prompt = f"""
            You are Max, a friendly content creation assistant. Just completed a tool step.
            
            TOOL COMPLETED: {tool_name}
            RESULT: {tool_result}
            CONTEXT: {context}
            
            Generate a conversational message (2-3 sentences) that:
            1. Acknowledges what was just completed with some specificity
            2. Shows enthusiasm about the progress
            3. Hints at what's coming next or why this step was important
            4. Uses friendly, professional language
            
            Avoid starting with "Great!" - be more varied and specific.
            """
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a friendly content creation assistant. Generate specific, conversational messages after tool completions that reference actual results."},
                    {"role": "user", "content": specific_prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )
            
            conversational_text = response.choices[0].message.content.strip()
            print(f"[DEBUG] Generated specific conversational response for {tool_name}: {conversational_text[:100]}...")
            
            return conversational_text
            
        except Exception as e:
            print(f"[DEBUG] Conversational response generation error: {e}")
            if tool_name == "Pitch Deck Information Extractor":
                return f"Perfect! I've extracted key details from your request including business information and presentation requirements. This gives us a solid foundation for your pitch deck. Now I'll fill in any missing pieces to make it complete."
    
            elif tool_name == "Smart Auto-Completion":
                return f"Excellent! I've intelligently completed the missing details for your pitch deck based on what you've shared. This ensures we have all the essential information needed for a compelling presentation. Ready to move on to the design phase!"
            
            elif tool_name == "Recent Generation Check":
                return f"I've checked our records and didn't find a recent presentation for this business. That's perfect - I'll create a brand new pitch deck tailored to your current needs. Let's get started on building something amazing!"
            
            elif tool_name == "Web Search Engine":
                return f"Excellent research completed! I found valuable insights and best practices that will help make your pitch deck more effective. These findings will guide our design decisions and ensure your presentation follows proven strategies."
            
            elif tool_name == "Slide Design Inspiration Finder":
                return f"Amazing design inspiration gathered! I found some fantastic visual examples that will make your pitch deck stand out. These creative ideas will help us create a presentation that's both professional and engaging."
            
            elif tool_name == "Slide Strategy Planning":
                return f"Strategic planning complete! I've mapped out the perfect structure for your pitch deck that will tell your story effectively. This thoughtful approach ensures every slide serves your presentation goals."
            
            elif tool_name == "AI Slides Generator":
                return f"Your pitch deck is ready! I've successfully created a professional presentation that incorporates all your information and follows best practices. You can now access and share your compelling pitch deck."
            
            else:
                # Generic fallback for other tools - make it more specific
                return f"I've successfully completed the {tool_name.lower()} step with great results. This important phase brings us closer to having a comprehensive pitch deck ready. Moving on to the next important phase now!"
        
    def save_rich_message(self, conversation_id: str, user_id: str, message_data: dict):
        """Save a rich message object to MongoDB"""
        try:
            # EXTRACT STANDARD FIELDS
            standard_data = {
                'conversation_id': conversation_id,
                'user_id': user_id,
                'sender': message_data.get('sender', 'agent'),
                'text': message_data.get('text', ''),
                'agent': self.agent_name
            }
            
            # EXTRACT RICH DATA (everything else)
            rich_data = {k: v for k, v in message_data.items() if k not in ['conversation_id', 'user_id', 'sender', 'text', 'agent', 'timestamp']}
            
            # SAVE USING NEW METHOD
            MongoDB.save_rich_message(**standard_data, **rich_data)
            
            # STORE IN PINECONE
            store_in_pinecone(
                agent_type=self.agent_name,
                role='agent',
                text=message_data.get('text', ''),
                user_id=user_id,
                conversation_id=conversation_id
            )
            
            return {"type": "success", "message": "Rich message saved successfully"}
        except Exception as e:
            print(f"Error saving rich message: {e}")
            return {"type": "error", "message": str(e)}

    async def stream_response(self, query: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream the complete response process"""
        try:
            all_tool_steps = []
            thinking_process_data = {}

            yield {
                "type": "tool_start",
                "tool_name": "Clarification",
                "message": "🔍 Clarifying the user requirement...",
                "status": "clarifying"
            }
         
            
            print(f"[DEBUG] Starting Super Agent response for: '{query}'")
            
            # Save user message to MongoDB
            if self.conversation_id and self.user_id:
                MongoDB.save_message(
                    conversation_id=self.conversation_id,
                    user_id=self.user_id,
                    sender='user',
                    text=query
                )
            
            # Store user query in Pinecone
            store_in_pinecone(
                agent_type=self.agent_name, 
                role="user", 
                text=query,
                user_id=self.user_id,
                conversation_id=self.conversation_id
            )
            
            # STEP 1: Thinking
            yield {
                "type": "thinking_start",
                "message": "🧠 Analyzing your request...",
                "status": "thinking_start"
            } 

            
            
            
            await asyncio.sleep(0.5)

            thinking_result = await self.get_real_model_thinking(query)
            print("Model thinking completed:", thinking_result)
            
            # ACCUMULATE: Store thinking process data
            thinking_process_data = {
                "thinking": thinking_result["thinking"],
                "reasoning": thinking_result["reasoning"],
                "analysis": thinking_result["analysis"],
                "plan": thinking_result["plan"]
            }
            
            yield {
                "type": "thinking_process",
                "message": "💭 Model's Real Thinking Process:",
                "thinking": thinking_result["thinking"],
                "reasoning": thinking_result["reasoning"],
                "analysis": thinking_result["analysis"],
                "plan": thinking_result["plan"],
                "status": "thinking_complete"
            }

            all_tool_steps.append({
                "type": "tool_result",
                "name": "Clarification",
                "message": f"Clarification completed, now moving on to next steps.....",
                "status": "completed",
                "conversationalText": "Clarification completed, now moving on to next steps.....",
                "timestamp": datetime.utcnow().isoformat()
            })   
            
            await asyncio.sleep(0.3)
            
            # STEP 2: Web Search
            yield {
                "type": "tool_start",
                "tool_name": "Web Research",
                "message": "🔍 Researching relevant information...",
                "status": "searching_web"
            }
            
            search_keywords = self.convert_to_keywords(query)
            search_results = self.search_with_keywords(search_keywords)
            formatted_results = self.format_search_results(search_results)

            conversational_text = await self.generate_conversational_response(
                tool_name="Web Search Engine",
                tool_result=f"Found {len(formatted_results)} relevant articles and references",
                context=f"Researching best practices for handeling user query"
            )
            all_tool_steps.append({
                "type": "tool_result",
                "name": "Web Search Engine",
                "message": f"✅ Found {len(formatted_results)} relevant articles and references",
                "status": "completed",
                "conversationalText": conversational_text,
                "data": {
                    "keywords": search_keywords,
                    "results": formatted_results
                },
                "timestamp": datetime.utcnow().isoformat()
            })
            
            yield {
                "type": "web_search_complete",
                "tool_name": "Web Research", 
                "message": f"✅ Found {len(formatted_results)} relevant articles",
                "data": {
                    "keywords": search_keywords,
                    "results": formatted_results
                },
                "status": "web_search_complete",
                "conversationalText": conversational_text
            }
            
            await asyncio.sleep(0.3)
            
            # STEP 3: Image Search (for inspiration)
            yield {
                "type": "tool_start",
                "tool_name": "Design Inspiration",
                "message": "🎨 Gathering visual inspiration...",
                "status": "searching_inspiration"
            }
            
            inspiration_images = self.search_images(query, num_results=8)

            # Generate conversational response for image search
            conversational_text = await self.generate_conversational_response(
                tool_name="Design Inspiration",
                tool_result=f"Found {len(inspiration_images)} design inspirations",
                context=f"Gathering visual inspiration for user"
            )

            all_tool_steps.append({
                "type": "tool_result",
                "name": "Design Inspiration",
                "message": f"🎨 Found {len(inspiration_images)} slide design inspirations",
                "status": "completed",
                "conversationalText": conversational_text,
                "data": inspiration_images,
                "timestamp": datetime.utcnow().isoformat()
            })    
            
            yield {
                "type": "inspiration_images",
                "tool_name": "Design Inspiration",
                "message": f"🎨 Found {len(inspiration_images)} design inspirations",
                "images": inspiration_images,
                "status": "inspiration_complete",
                "conversationalText": conversational_text,

            }
            
            await asyncio.sleep(0.3)
            
            # STEP 4: Generate Content Response
            yield {
                "type": "tool_start",
                "tool_name": "Content Generation",
                "message": "✨ Generating comprehensive content...",
                "status": "generating_content"
            }
            
            content_response = await self.generate_content_response(
                query=query,
                search_results=formatted_results,
                image_results=inspiration_images
            )

            conversational_text = await self.generate_conversational_response(
                tool_name="Content Generation",
                tool_result=f"Content Creation completed",
                context=f"This was last step where we generated the user content, now at this point, show excitement that we have generated, give some next suggestions based on user query: {query}"
            )

            all_tool_steps.append({
                "type": "tool_result",
                "name": "Content Generation",
                "message": f"Content Generation Completed",
                "status": "completed",
                "conversationalText": conversational_text,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            yield {
                "type": "tool_result",
                "tool_name": "Content Generation",
                "message": "✅ Content generated successfully",
                "status": "content_generated",
                "conversationalText": conversational_text,
            }
            
            await asyncio.sleep(0.3)
            
            # STEP 5: Stream Final Response
            yield {
                "type": "content_response",
                "message": content_response,
                "search_results": formatted_results,
                "inspiration_images": inspiration_images,
                "status": "complete"
            }
            
            # Save agent response to MongoDB
            if self.conversation_id and self.user_id:
                # Store in Pinecone
                store_in_pinecone(
                    agent_type=self.agent_name,
                    role='agent',
                    text=content_response,
                    user_id=self.user_id,
                    conversation_id=self.conversation_id
                )
            
            
            try:
                final_message_data = {
                    "sender": "agent",
                    "text": content_response,
                    "toolSteps": all_tool_steps,
                    "thinkingProcess": thinking_process_data,
                    "searchResults": {
                        "keywords": search_keywords,
                        "results": formatted_results
                    },
                    "inspirationImages": inspiration_images,
                    "agent": self.agent_name,
                    "status": "complete"
                }
                save_result = self.save_rich_message(self.conversation_id, self.user_id, final_message_data)
                if save_result["type"] == "success":
                    print(f"[DEBUG] 💾 Final message saved successfully")
            except Exception as save_error:
                print(f"[DEBUG] ❌ Error saving final message: {str(save_error)}")
                    
            
            print(f"[DEBUG] Super Agent response completed successfully")
            
        except Exception as e:
            print(f"[DEBUG] Super Agent error: {e}")
            yield {
                "type": "error",
                "message": f"I encountered an issue processing your request: {str(e)}",
                "status": "error"
            }

def get_super_agent(user_id: str = None, conversation_id: str = None):
    return SuperAgent(user_id, conversation_id)