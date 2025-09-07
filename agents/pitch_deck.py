import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool, AgentType
from core.config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENV, GROQ_API_KEY, SEARCHAPI_KEY, GAMMA_API_KEY
from core.database import MongoDB
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime
import asyncio
from typing import AsyncGenerator, Dict, Any
from groq import Groq
import re
import json
import requests
import time
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
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------
# Helper Functions
# ---------------------------
def embed_text(text: str):
    """Create an embedding for given text."""
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

def search_conversations_by_query(query: str, user_id: str, agent_type: str = "pitch-deck", top_k: int = 10):
    """Search conversations - get conversation_ids from Pinecone, conversation details from MongoDB"""
    try:
        print(f"[DEBUG] Searching conversations for: '{query}' (user: {user_id})")
        
        # Create embedding for search query
        query_vector = embed_text(query)
        
        # Search Pinecone - only get conversation_ids
        search_results = pinecone_index.query(
            vector=query_vector,
            top_k=top_k * 3,  # Get more results to find unique conversations
            include_metadata=True,
            include_values=False,
            filter={
                "agent_type": agent_type,
                "user_id": user_id
            }
        )
        
        print(f"[DEBUG] Found {len(search_results.matches)} vector matches")
        
        # Extract unique conversation_ids and keep best scores
        conversation_scores = {}  # conversation_id -> best_score
        
        for match in search_results.matches:
            if match.score < 0.2:  # Similarity threshold
                continue
                
            metadata = match.metadata or {}
            conv_id = metadata.get('conversation_id')
            
            if not conv_id:
                continue
            
            # Keep the best score for each conversation
            if conv_id not in conversation_scores or match.score > conversation_scores[conv_id]:
                conversation_scores[conv_id] = match.score
        
        print(f"[DEBUG] Found {len(conversation_scores)} unique conversations")
        
        # Get FULL conversation details from MongoDB
        search_results = []
        for conv_id, score in sorted(conversation_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            try:
                # Get conversation from MongoDB
                conversation = MongoDB.get_conversation_by_id(conv_id)
                if not conversation:
                    print(f"[DEBUG] Conversation {conv_id} not found in MongoDB")
                    continue
                
                # Verify ownership
                if conversation.get('userId') != user_id:  # Note: check field name
                    print(f"[DEBUG] Conversation {conv_id} belongs to different user")
                    continue
                
                # Add search score to conversation
                conversation['similarity_score'] = score
                search_results.append(conversation)
                
                print(f"[DEBUG] Added conversation: {conversation.get('title', 'Untitled')} (score: {score:.3f})")
                
            except Exception as e:
                print(f"[DEBUG] Error processing conversation {conv_id}: {e}")
                continue
        
        print(f"[DEBUG] Returning {len(search_results)} conversation results")
        return search_results
        
    except Exception as e:
        print(f"[DEBUG] Search error: {e}")
        return []

def generate_slides_gamma(info: dict):
    """Generate slides with Gamma AI and return proper format for frontend"""
    
    # Validate required information
    presentation_title = info.get('presentation_title', 'Pitch Deck')
    business_name = info.get('business_name', 'Business')
    target_audience = info.get('target_audience', 'investors')
    industry = info.get('industry', 'technology')
    problem_statement = info.get('problem_statement', '')
    solution = info.get('solution', '')
    business_model = info.get('business_model', '')
    market_size = info.get('market_size', '')
    funding_ask = info.get('funding_ask', '')
    
    # Create a detailed prompt for Gamma AI
    input_text = f"""Create a professional pitch deck for "{business_name}".

Title: {presentation_title}
Industry: {industry}
Target audience: {target_audience}

Problem: {problem_statement}
Solution: {solution}
Business Model: {business_model}
Market Size: {market_size}
Funding Ask: {funding_ask}

Include standard pitch deck sections like problem, solution, market size, business model, competition, team, and funding ask.
Make it visually appealing and professional for {target_audience}."""

    # Set up the API call to Gamma
    gamma_api_url = "https://public-api.gamma.app/v0.2/generations"
    
    # Headers with API key
    headers = {
        "Content-Type": "application/json",
        "X-API-KEY": GAMMA_API_KEY
    }
    
    # Prepare the request body
    request_body = {
        "inputText": input_text,
        "textMode": "condense",
        "format": "presentation",
        "themeName": "Oasis",  # Professional theme
        "numCards": 10,
        "cardSplit": "auto",
        "exportAs": "pptx",
        "textOptions": {
            "amount": "brief",
            "language": "en"
        },
        "imageOptions": {
            "source": "aiGenerated",
            "model": "imagen-4-pro",
            "style": "photorealistic"
        },
        "cardOptions": {
            "dimensions": "fluid"
        },
        "sharingOptions": {
            "workspaceAccess": "fullAccess",
            "externalAccess": "edit"
        }
    }
    
    try:
        print(f"[DEBUG] Generating slides with Gamma AI...")
        print(f"[DEBUG] Business: {business_name}")
        print(f"[DEBUG] Title: {presentation_title}")
        print(f"[DEBUG] Industry: {industry}")
        print(f"[DEBUG] Audience: {target_audience}")
        print(f"[DEBUG] Prompt length: {len(input_text)}")
        
        # Step 1: Make the initial API call to Gamma
        response = requests.post(gamma_api_url, json=request_body, headers=headers, timeout=120)
        
        # Check for HTTP errors
        if response.status_code not in [200, 201]:  # Accept both 200 and 201 as success
            error_msg = f"Gamma API returned status code {response.status_code}: {response.text}"
            print(f"[DEBUG] {error_msg}")
            return {
                "type": "error",
                "message": f"I encountered an issue generating your slides: The presentation service returned an error. Please try again with different specifications.",
                "slides_url": None,
                "business_info": info
            }
            
        # Parse response
        response_data = response.json()
        print(f"[DEBUG] Initial response data: {response_data}")
        
        # Check for generation ID
        if "generationId" not in response_data:
            error_msg = "No generation ID returned from Gamma API"
            print(f"[DEBUG] Gamma API error: {error_msg}")
            return {
                "type": "error",
                "message": f"I encountered an issue generating your slides: The presentation service didn't return a valid generation ID. Please try again later.",
                "slides_url": None,
                "business_info": info
            }
        
        # Step 2: Get the generation ID
        generation_id = response_data["generationId"]
        
        # Step 3: Poll the status endpoint until generation is complete
        status_url = f"https://public-api.gamma.app/v0.2/generations/{generation_id}"
        
        print(f"[DEBUG] Generation started. ID: {generation_id}")
        print(f"[DEBUG] Polling status URL: {status_url}")
        
        # Set up polling parameters
        max_attempts = 12  # Maximum number of polling attempts
        poll_interval = 5  # Time between polling attempts in seconds
        attempts = 0
        
        gamma_url = f"https://gamma.app/view/{generation_id}"  # Default view URL
        export_url = None  # Will hold the downloadable PPTX URL when available
        
        # Loop until generation is complete or max attempts reached
        while attempts < max_attempts:
            attempts += 1
            print(f"[DEBUG] Polling attempt {attempts}/{max_attempts}...")
            
            # Make the status request
            status_response = requests.get(status_url, headers=headers, timeout=30)
            
            if status_response.status_code != 200:
                print(f"[DEBUG] Status check failed with code {status_response.status_code}: {status_response.text}")
                if attempts == max_attempts:
                    return {
                        "type": "error",
                        "message": f"I'm having trouble checking the status of your presentation. You can try viewing it directly at {gamma_url}",
                        "slides_url": gamma_url,
                        "business_info": info
                    }
                continue  # Try again
                
            # Parse status response
            status_data = status_response.json()
            print(f"[DEBUG] Status data: {status_data}")
            
            # Check if generation is complete
            status = status_data.get("status", "").lower()
            print(f"[DEBUG] Current status: {status}")
            
            if status == "completed":
                # Get the export URL if available
                export_url = status_data.get("exportUrl")
                if export_url:
                    print(f"[DEBUG] Export URL found: {export_url}")
                break
            
            elif status == "failed":
                error_message = status_data.get("errorMessage", "Unknown error")
                print(f"[DEBUG] Generation failed: {error_message}")
                return {
                    "type": "error",
                    "message": f"I encountered an issue generating your slides: {error_message}. Please try again with different specifications.",
                    "slides_url": None,
                    "business_info": info
                }
            
            elif status == "pending" or status == "running":
                print(f"[DEBUG] Generation still in progress, status: {status}")
                # Wait before polling again
                time.sleep(poll_interval)
                continue
                
            else:
                print(f"[DEBUG] Unknown status: {status}")
                if attempts == max_attempts:
                    break
                time.sleep(poll_interval)
        
        # Check if we've hit max attempts without completion
        if attempts >= max_attempts and status != "completed":
            print("[DEBUG] Maximum polling attempts reached without completion")
            return {
                "type": "partial_success",
                "message": f"Your presentation is still being generated. You can view it at {gamma_url} when it's ready (it may take a few more moments to complete).",
                "slides_url": gamma_url,
                "business_info": info
            }
            
        # Success! Create success message
        success_message = f"🎉 **Your {presentation_title} pitch deck for {business_name} is ready!**\n\nI've created a professional pitch deck targeting {target_audience} in the {industry} industry. Click the link below to access your presentation."
        
        # We'll use the export URL if available, otherwise the view URL
        final_url = export_url or gamma_url
        
        print(f"[DEBUG] Successfully generated slides!")
        print(f"[DEBUG] Final URL: {final_url}")
        
        # Return successful result
        return {
            "type": "slides_generated",
            "message": success_message,
            "slides_url": final_url,
            "export_url": export_url,  # Include export URL separately in case frontend wants to offer download option
            "view_url": gamma_url,     # Include view URL separately for online viewing
            "business_info": {
                "business_name": business_name,
                "presentation_title": presentation_title,
                "target_audience": target_audience,
                "industry": industry,
                "problem_statement": problem_statement,
                "solution": solution,
                "business_model": business_model,
                "market_size": market_size,
                "funding_ask": funding_ask
            }
        }
    
    except Exception as e:
        print(f"[DEBUG] Gamma AI generation error: {str(e)}")
        
        # Return error result
        return {
            "type": "error",
            "message": f"I encountered an issue generating your slides: {str(e)}. Let me try again with different specifications, or you can adjust your requirements.",
            "slides_url": None,
            "business_info": info
        }
# ---------------------------
# Pitch Deck Agent with MongoDB
# ---------------------------
class PitchDeckAgent:
    def __init__(self, user_id: str = None, conversation_id: str = None):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.agent_name = "pitch-deck"
        self.last_generated_slides = None
        self.groq_client = groq_client
        self.reasoning_model = "deepseek-r1-distill-llama-70b"
        self.show_thinking = True  # Enable thinking display
        self.searchapi_key = SEARCHAPI_KEY
        self.search_base_url = "https://www.searchapi.io/api/v1/search"
        self.detected_deck_type = None
        self.user_context = None
        print(f"[DEBUG] Initialized with reasoning model: {self.reasoning_model}")
        
        # Enhanced pitch deck info
        self.pitch_info = {
            "business_name": None,
            "presentation_title": None,
            "target_audience": None,
            "industry": None,
            "problem_statement": None,
            "solution": None,
            "business_model": None,
            "market_size": None,
            "funding_ask": None,
        }

        # Add the system prompt as an attribute
        self.system_prompt = """You are Max, a professional pitch deck creator who helps entrepreneurs build compelling pitch decks for investors, clients, and stakeholders.

Your capabilities:
📊 **Pitch Deck Creation:**
- Startup investor presentations
- Client proposal decks
- Product launch presentations
- Business overview decks
- Marketing strategy presentations

🎯 **Presentation Consultation:**
- Storytelling and narrative flow
- Slide structure and organization
- Presentation design principles
- Content optimization for target audience
- Effective data visualization

When users request any pitch deck, immediately use the Generate_Pitch_Deck tool.

Key guidelines:
- Focus on clear, concise messaging
- Maintain professional design principles
- Structure slides in a logical narrative flow
- Ask clarifying questions when needed

Always prioritize using the tool for any pitch deck requests."""

        self.llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Load conversation history if conversation_id exists
        if self.conversation_id and self.user_id:
            self.load_conversation_history()

        # Create the pitch deck generation tool
        tools = [
            Tool(
                name="Generate_Pitch_Deck",
                func=self.smart_deck_generator,
                description="""Use this tool IMMEDIATELY when the user requests a pitch deck including:
                
                INVESTOR DECKS: startup pitch, investor presentation, funding pitch
                CLIENT PRESENTATIONS: client proposal, sales pitch, business presentation
                PRODUCT PRESENTATIONS: product launch, feature overview, roadmap presentation
                
                This tool handles information collection AND actual slide generation for ALL pitch deck requests.
                Always use this tool for any pitch deck generation requests."""
            )
        ]

        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            agent_kwargs={
                'system_message': """You are Max, a professional pitch deck creator who helps users create compelling presentations.

Your capabilities:
- Pitch deck creation using Gamma AI
- Investor presentations for startups
- Client proposal presentations
- Product launch decks
- Business overview presentations

When users mention wanting ANY pitch deck, immediately use the Generate_Pitch_Deck tool.

CRITICAL INSTRUCTION: When the tool returns a response that starts with "SLIDES_GENERATED|", you MUST return that EXACT response without any modifications, additions, or formatting changes. Do not convert it to markdown, do not add explanations, just return the exact string as-is.

Examples:
- If tool returns: "SLIDES_GENERATED|https://slides-url|message"
- You return: "SLIDES_GENERATED|https://slides-url|message" (EXACTLY)

For all other responses, be conversational and friendly.

Key guidelines:
- Always use the tool for ANY pitch deck-related requests
- NEVER modify SLIDES_GENERATED responses
- Trust the tool to handle information collection and generation
- Focus on being helpful and strategic for non-deck conversations

Always prioritize using the tool over giving generic advice."""
            }
        )

    def extract_pitch_info_from_conversation(self, messages):
        """Use GPT to intelligently extract pitch deck information from conversation"""
        
        # Build FULL conversation text - ALL messages, not just recent
        conversation_text = ""
        for msg in messages:
            role = "User" if msg['sender'] == 'user' else "Assistant"
            conversation_text += f"{role}: {msg['text']}\n"
        
        if not conversation_text.strip():
            print("[DEBUG] No conversation history to extract from")
            return
        
        print(f"[DEBUG] Extracting from FULL conversation: {conversation_text}")
        
        # Enhanced extraction
        extraction_prompt = f"""
        Analyze this COMPLETE conversation and extract pitch deck information.
        
        Conversation:
        {conversation_text}

        Extract the MOST RECENT pitch deck information:
        - business_name: The company or startup name
        - presentation_title: Title of the pitch deck
        - target_audience: Who the pitch is for (investors, clients, etc.)
        - industry: The industry the business operates in
        - problem_statement: The problem being solved
        - solution: The proposed solution
        - business_model: How the business makes money
        - market_size: Target market information
        - funding_ask: Amount of funding requested (if applicable)

        Return ONLY valid JSON:
        {{"business_name": "most recent business name or null", "presentation_title": "title or null", "target_audience": "audience or null", "industry": "industry or null", "problem_statement": "problem or null", "solution": "solution or null", "business_model": "business model or null", "market_size": "market size or null", "funding_ask": "funding ask or null"}}
        """
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Extract pitch deck information from conversations. Return ONLY valid JSON."},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            extracted_text = response.choices[0].message.content.strip()
            print(f"[DEBUG] GPT extraction result: {extracted_text}")
            
            # Clean up markdown
            if extracted_text.startswith('```json'):
                extracted_text = extracted_text.replace('```json', '').replace('```', '').strip()
            elif extracted_text.startswith('```'):
                extracted_text = extracted_text.replace('```', '').strip()
            
            import json
            extracted_info = json.loads(extracted_text)
            
            # Update with new information
            for key, value in extracted_info.items():
                if key in self.pitch_info and value and value.lower() not in ["null", "", "none"]:
                    self.pitch_info[key] = value
                    print(f"[DEBUG] Updated {key}: {value}")
            
            print(f"[DEBUG] Final pitch_info: {self.pitch_info}")
            
        except Exception as e:
            print(f"[DEBUG] Extraction error: {e}")

    def intelligent_auto_complete(self, provided_info: dict, deck_type: str = "investor"):
        """Enhanced auto-completion with deck-type awareness"""
        
        known_info = ""
        missing_fields = []
        for key, value in provided_info.items():
            if value:
                known_info += f"{key}: {value}\n"
            else:
                missing_fields.append(key)
        
        if not known_info.strip():
            return provided_info
        
        # Create completion prompt
        business_name = provided_info.get('business_name', 'Business')
        
        completion_prompt = f"""
        Based on the following pitch deck information, intelligently suggest appropriate values for the missing fields.
        The user wants to create a {deck_type} pitch deck.

        Known information:
        {known_info}

        Please complete this JSON with intelligent defaults for the missing fields: {missing_fields}

        Guidelines based on deck type "{deck_type}":
        - For investor decks: Focus on market opportunity, traction, and funding needs
        - For client presentations: Emphasize solutions, benefits, and implementation
        - For product launches: Highlight features, benefits, and roadmap
        
        Industry-specific suggestions:
        - Tech: Emphasize scalability, innovation, and market disruption
        - Healthcare: Focus on patient outcomes, regulatory approval, and market need
        - Retail/Consumer: Highlight brand positioning, customer acquisition, and market trends
        - Finance: Emphasize risk management, compliance, and profitability

        Return ONLY a complete JSON object with all fields.
        """
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert pitch deck consultant. Always return valid JSON."},
                    {"role": "user", "content": completion_prompt}
                ],
                temperature=0.3,
                max_tokens=600
            )
            
            completed_text = response.choices[0].message.content.strip()
            print(f"[DEBUG] Auto-completion result: {completed_text}")
            
            # Clean up any markdown formatting
            if completed_text.startswith('```json'):
                completed_text = completed_text.replace('```json', '').replace('```', '').strip()
            elif completed_text.startswith('```'):
                completed_text = completed_text.replace('```', '').strip()
            
            import json
            completed_info = json.loads(completed_text)
            
            # Merge with original info
            final_info = {}
            for key in provided_info.keys():
                final_info[key] = provided_info[key] if provided_info[key] else completed_info.get(key)
            
            print(f"[DEBUG] Final completed info: {final_info}")
            return final_info
            
        except Exception as e:
            print(f"[DEBUG] Auto-completion error: {e}")
            return provided_info

    # Search functionality methods
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
        """Convert user query to search keywords using GPT"""
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Convert the user query into a keyword optimized for search engine query to find pitch deck examples/help/guidance. Return single keyword only."},
                    {"role": "user", "content": f"User wants: {query}. Extract pitch deck-related search keyword."}
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
        """Search for pitch deck inspiration images"""
        print(f"[DEBUG] Starting image search for: '{query}' with {num_results} results")
        
        # Search queries for relevant platforms
        pitch_query = f"pitch deck examples {query}"
        presentation_query = f"presentation design {query}"
        
        print(f"[DEBUG] Pitch query: {pitch_query}")
        print(f"[DEBUG] Presentation query: {presentation_query}")
        
        all_images = []
        
        # Search pitch decks
        try:
            params = {
                "engine": "google_images",
                "q": pitch_query,
                "api_key": self.searchapi_key,
                "num": 5
            }
            
            response = requests.get(self.search_base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                pitch_data = response.json()
                
                images_list = pitch_data.get("images", [])
                print(f"[DEBUG] Pitch deck total images available: {len(images_list)}")
                
                for i, item in enumerate(images_list[:5]):
                    
                    original_data = item.get("original", {})
                    source_data = item.get("source", {})
                    
                    image_data = {
                        "title": item.get("title", "Pitch Deck Example")[:100],
                        "original": original_data.get("link") if isinstance(original_data, dict) else original_data,
                        "thumbnail": item.get("thumbnail"),
                        "link": source_data.get("link") if isinstance(source_data, dict) else None,
                        "source": "Pitch Deck Example",
                        "position": item.get("position", i+1)
                    }
                    
                    if image_data["original"] and image_data["thumbnail"]:
                        all_images.append(image_data)
                        
            else:
                print(f"[DEBUG] Pitch search API error: {response.status_code}")
                
        except Exception as e:
            print(f"[DEBUG] Pitch search error: {e}")

        # Search presentation designs
        try:
            params = {
                "engine": "google_images", 
                "q": presentation_query,
                "api_key": self.searchapi_key,
                "num": 5
            }
            
            response = requests.get(self.search_base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                pres_data = response.json()
                
                images_list = pres_data.get("images", [])
                
                for i, item in enumerate(images_list[:5]):
                    
                    original_data = item.get("original", {})
                    source_data = item.get("source", {})
                    
                    image_data = {
                        "title": item.get("title", "Presentation Design")[:100],
                        "original": original_data.get("link") if isinstance(original_data, dict) else original_data,
                        "thumbnail": item.get("thumbnail"),
                        "link": source_data.get("link") if isinstance(source_data, dict) else None,
                        "source": "Presentation Design",
                        "position": item.get("position", i+1)
                    }
                    
                    if image_data["original"] and image_data["thumbnail"]:
                        all_images.append(image_data)
                        
        except Exception as e:
            print(f"[DEBUG] Presentation search error: {e}")

        print(f"[DEBUG] Final result: {len(all_images)} design inspiration images collected")
        return all_images

    def format_search_results(self, results):
        """Format web search results for display in frontend Sources Modal"""
        articles = []
        for item in results.get("organic_results", []):
            articles.append({
                "title": item.get("title"),
                "link": item.get("link"), 
                "source": item.get("source"),
                "snippet": item.get("snippet")
            })
        return articles[:8]

    def extract_from_current_input(self, user_input: str):
        """Extract pitch deck info from current input"""
        
        current_info_text = ""
        if any(v for v in self.pitch_info.values()):
            current_info_text = f"Currently working on:\n"
            for key, value in self.pitch_info.items():
                if value:
                    current_info_text += f"- {key}: {value}\n"
            current_info_text += "\n"
        
        extraction_prompt = f"""
        {current_info_text}User input: "{user_input}"
        
        Extract pitch deck information and return ONLY valid JSON, no explanations.
        """
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "CRITICAL: Return ONLY valid JSON with no explanations, no text before or after. Just the JSON object."},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            extracted_text = response.choices[0].message.content.strip()
            print(f"[DEBUG] Current input extraction: {extracted_text}")
            
            # Clean up response
            if "```json" in extracted_text:
                extracted_text = extracted_text.split("```json")[1].split("```")[0].strip()
            elif "```" in extracted_text:
                extracted_text = extracted_text.split("```")[1].split("```")[0].strip()
            
            # Find JSON object in response
            import re
            json_match = re.search(r'\{[^}]+\}', extracted_text)
            if json_match:
                extracted_text = json_match.group(0)
            
            import json
            extracted_info = json.loads(extracted_text)
            
            # Update pitch_info with new extracted information
            for key, value in extracted_info.items():
                if key in self.pitch_info and value and value.lower() not in ["null", "", "none"]:
                    self.pitch_info[key] = value
                    print(f"[DEBUG] Updated from current input - {key}: {value}")
            
            # Store the deck type separately if detected
            if extracted_info.get("deck_type"):
                self.detected_deck_type = extracted_info["deck_type"]
                print(f"[DEBUG] Detected deck type from current input: {self.detected_deck_type}")
            
            print(f"[DEBUG] Final pitch_info after current input: {self.pitch_info}")
                    
        except Exception as e:
            print(f"[DEBUG] Current input extraction error: {e}")
            
            # Simple fallback - extract basic keywords
            user_lower = user_input.lower()
            
            if "investor" in user_lower or "funding" in user_lower:
                self.detected_deck_type = "investor_pitch"
            elif "client" in user_lower or "proposal" in user_lower:
                self.detected_deck_type = "client_proposal"
            elif "product" in user_lower or "launch" in user_lower:
                self.detected_deck_type = "product_launch"
            elif any(word in user_lower for word in ["pitch", "deck", "presentation"]):
                self.detected_deck_type = "general_pitch"

    def load_pitch_info(self):
        """Load pitch deck info from user data"""
        if not self.user_id:
            return
            
        try:
            pitch_info = MongoDB.get_user_pitch_info(self.user_id)
            
            # Update pitch_info with saved data
            for key, value in pitch_info.items():
                if key in self.pitch_info and value:
                    self.pitch_info[key] = value
                    print(f"[DEBUG] Loaded {key}: {value}")
            
            print(f"[DEBUG] Loaded pitch info: {self.pitch_info}")
                
        except Exception as e:
            print(f"[DEBUG] Error loading pitch info: {e}")

    def save_pitch_info(self):
        """Save current pitch_info to user data"""
        if not self.user_id:
            return
            
        try:
            # Filter out None values and add timestamp
            pitch_info_data = {k: v for k, v in self.pitch_info.items() if v is not None}
            pitch_info_data["lastUpdated"] = datetime.utcnow().isoformat()
            
            success = MongoDB.update_user_pitch_info(self.user_id, pitch_info_data)
            
            if success:
                print(f"[DEBUG] Saved pitch info: {pitch_info_data}")
            else:
                print("[DEBUG] Failed to save pitch info")
                
        except Exception as e:
            print(f"[DEBUG] Error saving pitch info: {e}")

    def detect_generation_intent(self, user_input: str) -> bool:
        """Enhanced generation intent detection"""
        
        intent_prompt = f"""
        Analyze this user message and determine if they want to generate/create a pitch deck NOW.
        
        User message: "{user_input}"
        
        Return ONLY "YES" if they want to generate a pitch deck now, or "NO" if they're just providing information.
        
        Examples that mean YES (generate):
        - "create pitch deck" → YES
        - "generate investor presentation" → YES  
        - "make slides for my pitch" → YES
        - "build a pitch deck" → YES
        - "create it now" → YES
        
        Examples that mean NO (just providing info):
        - "my business name is Acme" → NO
        - "our target market is enterprise" → NO
        - "what should I include in my pitch?" → NO
        
        Answer: """
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Detect if user wants to generate/create a pitch deck NOW. Always respond with only YES or NO."},
                    {"role": "user", "content": intent_prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            intent = response.choices[0].message.content.strip().upper()
            print(f"[DEBUG] Generation intent detected: {intent} for input: '{user_input}'")
            
            return intent == "YES"
            
        except Exception as e:
            print(f"[DEBUG] Intent detection error: {e}")
            # Fallback keywords for pitch decks
            generation_keywords = [
                "generate", "create", "make", "build", "design",
                "pitch deck", "slide deck", "presentation", "slides", 
                "investor deck", "client pitch", "product presentation"
            ]
            
            return any(phrase in user_input.lower() for phrase in generation_keywords)

    def load_conversation_history(self):
        """Load previous messages into memory"""
        print(f"Getting messages of userID {self.user_id} from conversation {self.conversation_id}")
        messages = MongoDB.get_conversation_messages(self.conversation_id, self.user_id)
        for msg in messages:
            if msg['sender'] == 'user':
                self.memory.chat_memory.add_user_message(msg['text'])
            else:
                self.memory.chat_memory.add_ai_message(msg['text'])

        print(f"[DEBUG] Loaded {len(messages)} messages from conversation history")


    def smart_deck_generator(self, user_input: str = "") -> str:
        """Smart pitch deck generation"""
        
        print(f"[DEBUG] smart_deck_generator called with input: '{user_input}'")
        print(f"[DEBUG] Initial pitch_info: {self.pitch_info}")
        
        self.user_context = user_input
        self.detected_deck_type = None
        conversation_detected_type = None
        
        # Process input
        processed_input = user_input
        if user_input.startswith('{"') and user_input.endswith('}'):
            try:
                import json
                parsed_input = json.loads(user_input)
                processed_input = parsed_input.get("description", "generate pitch deck")
            except:
                pass
        
        # Get ALL conversation messages
        recent_messages = []
        if self.conversation_id and self.user_id:
            messages = MongoDB.get_conversation_messages(self.conversation_id, self.user_id)
            recent_messages = messages
            print(f"[DEBUG] Found {len(recent_messages)} total messages for analysis")
        
        # Extract information from FULL conversation
        if recent_messages:
            self.extract_pitch_info_from_conversation(recent_messages)
            conversation_detected_type = getattr(self, 'detected_deck_type', None)
            print(f"[DEBUG] Conversation detected type: {conversation_detected_type}")
        
        # Extract from current input
        if processed_input and processed_input.strip():
            self.extract_from_current_input(processed_input)
            current_input_detected_type = getattr(self, 'detected_deck_type', None)
            print(f"[DEBUG] Current input detected type: {current_input_detected_type}")
        
        # Determine deck type from conversation or current input
        deck_type = current_input_detected_type or conversation_detected_type or "investor"
        print(f"[DEBUG] Final deck type: {deck_type}")
        
        # Check if we have enough information to generate a deck
        missing_required_fields = []
        required_fields = ["business_name", "target_audience", "industry"]
        
        for field in required_fields:
            if not self.pitch_info.get(field):
                missing_required_fields.append(field)
        
        # If missing required fields, auto-complete or ask questions
        if missing_required_fields:
            print(f"[DEBUG] Missing required fields: {missing_required_fields}")
            
            # Try auto-complete for missing fields
            auto_completed = self.intelligent_auto_complete(self.pitch_info.copy(), deck_type)
            still_missing = []
            
            for field in missing_required_fields:
                if auto_completed.get(field):
                    self.pitch_info[field] = auto_completed[field]
                    print(f"[DEBUG] Auto-completed {field}: {auto_completed[field]}")
                else:
                    still_missing.append(field)
            
            # If still missing after auto-complete, ask questions
            if still_missing:
                print(f"[DEBUG] Still missing after auto-complete: {still_missing}")
                comprehensive_questions = self.ask_comprehensive_pitch_questions()
                
                # Save the current state to DB for future completion
                self.save_pitch_info()
                
                return comprehensive_questions
        
        # We have all required information, proceed with generation
        print(f"[DEBUG] All required fields present, generating deck")
        print(f"[DEBUG] Final pitch_info: {self.pitch_info}")
        
        # Save to DB before generation
        self.save_pitch_info()
        
        # Generate pitch deck with Gamma AI
        result = generate_slides_gamma(self.pitch_info)
        
        # Save the generated URL
        if result["type"] == "slides_generated":
            self.last_generated_slides = result["slides_url"]
            
            # Format for agent output
            message = (
                f"SLIDES_GENERATED|{result['slides_url']}|{result['message']}"
            )
            
            print(f"[DEBUG] Slides generated successfully: {result['slides_url']}")
            return message
        else:
            # Error in generation
            error_message = f"I encountered an issue when creating your pitch deck: {result.get('message', 'Unknown error')}"
            print(f"[DEBUG] Generation error: {error_message}")
            return error_message


    async def handle_query_stream(self, query: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle user query with streaming responses"""
        
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
            agent_type="pitch-deck", 
            role="user", 
            text=query,
            user_id=self.user_id,
            conversation_id=self.conversation_id
        )

        try:
            # Stream the agent processing
            async for chunk in self.stream_agent_response(query):
                yield chunk
                
        except Exception as e:
            print(f"Streaming error: {e}")
            yield {
                "type": "error",
                "message": "I'm experiencing technical difficulties. Please try again in a moment."
            }

    async def stream_agent_response(self, query: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream agent response with tool usage visibility"""
        
        try:
            # Check if this is a pitch deck generation request
            wants_generation = self.detect_generation_intent(query)
            
            if wants_generation:
                # Use real thinking-enabled slide generation
                async for chunk in self.stream_slides_generation_with_real_thinking(query):
                    yield chunk
            else:
                # Stream regular conversation
                async for chunk in self.stream_conversation_response(query):
                    yield chunk
                    
        except Exception as e:
            yield {
                "type": "error", 
                "message": f"Processing error: {str(e)}"
            }

    async def stream_slides_generation_with_real_thinking(self, query: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream slides generation with REAL model thinking - TRUE SEQUENTIAL EXECUTION"""
        try:
            # Initialize: Accumulate tool steps and thinking process data
            all_tool_steps = []
            thinking_process_data = {}

            # Save user message
            if self.conversation_id and self.user_id:
                MongoDB.save_message(
                    conversation_id=self.conversation_id,
                    user_id=self.user_id,
                    sender='user',
                    text=query
                )
                print(f"[DEBUG] User message saved to MongoDB: {query}")
            
            # Store user query in Pinecone
            store_in_pinecone(
                agent_type="pitch-deck", 
                role="user", 
                text=query,
                user_id=self.user_id,
                conversation_id=self.conversation_id
            )
            print(f"[DEBUG] User message stored in Pinecone")
            
            # STEP 1: REAL MODEL THINKING ABOUT THE REQUEST
            yield {
                "type": "thinking_start",
                "message": "🧠 Thinking...",
                "status": "thinking"
            }
            
            # WAIT FOR ACTUAL THINKING TO COMPLETE
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
            
            await asyncio.sleep(0.3)

            # STEP 2: WEB SEARCH
            yield {
                "type": "tool_start",
                "tool_name": "Web Search Engine",
                "message": "🔍 Searching for pitch deck examples and best practices...",
                "status": "searching_web"
            }
            
            # WAIT FOR ACTUAL SEARCH TO COMPLETE
            search_keywords = self.convert_to_keywords(query)
            search_results = self.search_with_keywords(search_keywords)
            formatted_results = self.format_search_results(search_results)
            
            # ACCUMULATE: Add tool step
            all_tool_steps.append({
                "type": "tool_result",
                "name": "Web Search Engine",
                "message": f"✅ Found {len(formatted_results)} relevant articles and references",
                "status": "completed",
                "data": {
                    "keywords": search_keywords,
                    "results": formatted_results
                },
                "timestamp": datetime.utcnow().isoformat()
            })
            
            yield {
                "type": "web_search_complete",
                "tool_name": "Web Search Engine", 
                "message": f"✅ Found {len(formatted_results)} relevant articles and references",
                "data": {
                    "keywords": search_keywords,
                    "results": formatted_results
                },
                "status": "web_search_complete"
            }
            
            await asyncio.sleep(0.3)
            
            # STEP 3: IMAGE SEARCH
            yield {
                "type": "tool_start",
                "tool_name": "Slide Design Inspiration Finder",
                "message": "🎨 Searching for pitch deck design inspiration...",
                "status": "searching_inspiration"
            }
            
            # WAIT FOR ACTUAL IMAGE SEARCH TO COMPLETE
            inspiration_images = self.search_images(f"{search_keywords} pitch deck slides", num_results=8)
            
            # ACCUMULATE: Add tool step
            all_tool_steps.append({
                "type": "tool_result",
                "name": "Slide Design Inspiration Finder",
                "message": f"🎨 Found {len(inspiration_images)} slide design inspirations",
                "status": "completed",
                "data": inspiration_images,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            yield {
                "type": "inspiration_images",
                "tool_name": "Slide Design Inspiration Finder",
                "message": f"🎨 Found {len(inspiration_images)} slide design inspirations",
                "images": inspiration_images,
                "status": "inspiration_complete"
            }
            
            await asyncio.sleep(0.3)
            
            # STEP 4: PITCH DECK INFORMATION EXTRACTION
            yield {
                "type": "tool_start",
                "tool_name": "Pitch Deck Information Extractor",
                "message": "📋 Extracting pitch deck information from your request...",
                "status": "extracting_info"
            }
            
            # WAIT FOR ACTUAL EXTRACTION THINKING TO COMPLETE
            extraction_thinking = await self.get_real_extraction_thinking(query)
            
            # ACCUMULATE: Update thinking process
            thinking_process_data.update({
                "process": extraction_thinking["process"],
                "findings": extraction_thinking["findings"]
            })
            
            yield {
                "type": "thinking_process",
                "message": "🧠 Information Extraction Analysis:",
                "thinking": extraction_thinking["thinking"],
                "process": extraction_thinking["process"],
                "findings": extraction_thinking["findings"],
                "status": "extraction_thinking"
            }
            
            # PERFORM ACTUAL EXTRACTION
            recent_messages = []
            if self.conversation_id and self.user_id:
                recent_messages = MongoDB.get_conversation_messages(self.conversation_id, self.user_id)
            
            if recent_messages:
                self.extract_pitch_info_from_conversation(recent_messages)
            
            if query and query.strip():
                self.extract_from_current_input(query)
            
            extracted_info = {k: v for k, v in self.pitch_info.items() if v}
            
            # ACCUMULATE: Add tool step
            all_tool_steps.append({
                "type": "tool_result",
                "name": "Pitch Deck Information Extractor",
                "message": f"✅ Extracted: {', '.join(extracted_info.keys()) if extracted_info else 'Basic information'}",
                "status": "completed",
                "data": extracted_info,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            yield {
                "type": "tool_result",
                "tool_name": "Pitch Deck Information Extractor",
                "message": f"✅ Extracted: {', '.join(extracted_info.keys()) if extracted_info else 'Basic information'}",
                "data": extracted_info,
                "status": "info_extracted"
            }
            
            # Check if we need more info
            if not self.pitch_info.get("business_name"):
                comprehensive_questions = self.ask_comprehensive_pitch_questions()
                
                yield {
                    "type": "complete",
                    "status": "awaiting_input",
                    "message": comprehensive_questions,
                    "final_data": {
                        "search_results": formatted_results,
                        "search_keywords": search_keywords,
                        "inspiration_images": inspiration_images,
                        "tool_steps": all_tool_steps,
                        "thinking_process": thinking_process_data
                    }
                }
                return
            
            await asyncio.sleep(0.3)
            
            # STEP 5: AUTO-COMPLETION
            missing_info = [k for k, v in self.pitch_info.items() if not v]
            
            if missing_info:
                yield {
                    "type": "tool_start",
                    "tool_name": "Smart Auto-Completion",
                    "message": "🧠 Smart-completing missing pitch deck details...",
                    "status": "auto_completing"
                }
                
                # WAIT FOR ACTUAL AUTO-COMPLETION TO COMPLETE
                deck_type = self.detected_deck_type or "investor"
                auto_completed = self.intelligent_auto_complete(self.pitch_info.copy(), deck_type)
                for key, value in auto_completed.items():
                    if not self.pitch_info.get(key):
                        self.pitch_info[key] = value
                
                completed_fields = [k for k in missing_info if self.pitch_info.get(k)]
                
                # ACCUMULATE: Add tool step
                all_tool_steps.append({
                    "type": "tool_result",
                    "name": "Smart Auto-Completion",
                    "message": f"✅ Completed: {', '.join(completed_fields) if completed_fields else 'Pitch deck information'}",
                    "status": "completed",
                    "data": {k: self.pitch_info[k] for k in completed_fields if self.pitch_info.get(k)},
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                yield {
                    "type": "tool_result",
                    "tool_name": "Smart Auto-Completion", 
                    "message": f"✅ Completed: {', '.join(completed_fields) if completed_fields else 'Pitch deck information'}",
                    "data": {k: self.pitch_info[k] for k in completed_fields if self.pitch_info.get(k)},
                    "status": "auto_completed"
                }
            
            await asyncio.sleep(0.3)
            
            # STEP 6: SLIDE DESIGN THINKING
            yield {
                "type": "tool_start",
                "tool_name": "Slide Strategy Planning",
                "message": "📊 Planning slide structure and presentation strategy...",
                "status": "slide_thinking"
            }
            
            # WAIT FOR ACTUAL SLIDE THINKING TO COMPLETE
            slide_thinking = await self.get_real_slide_thinking(self.pitch_info, query)
            
            # ACCUMULATE: Update thinking process
            thinking_process_data.update({
                "slide_strategy": slide_thinking["slide_strategy"],
                "key_slides": slide_thinking["key_slides"],
                "narrative_flow": slide_thinking["narrative_flow"]
            })
            
            yield {
                "type": "thinking_process",
                "message": "📊 Slide Strategy Planning:",
                "thinking": slide_thinking["thinking"],
                "slide_strategy": slide_thinking["slide_strategy"],
                "key_slides": slide_thinking["key_slides"],
                "narrative_flow": slide_thinking["narrative_flow"],
                "status": "slide_thinking_complete"
            }

            # MARK TOOL AS COMPLETED
            all_tool_steps.append({
                "type": "tool_result",
                "name": "Slide Strategy Planning",
                "message": "✅ Presentation strategy and slide structure planned",
                "status": "completed",
                "data": {
                    "slide_strategy": slide_thinking["slide_strategy"],
                    "key_slides": slide_thinking["key_slides"]
                },
                "timestamp": datetime.utcnow().isoformat()
            })

            yield {
                "type": "tool_result",
                "tool_name": "Slide Strategy Planning",
                "message": "✅ Presentation strategy and slide structure planned",
                "status": "completed"
            }
            
            await asyncio.sleep(0.3)
            
            # STEP 7: SLIDES GENERATION
            yield {
                "type": "tool_start",
                "tool_name": "Gamma AI Slides Generator",
                "message": "✨ Generating your pitch deck with Gamma AI...",
                "status": "generating_slides"
            }
            
            # WAIT FOR ACTUAL SLIDES GENERATION TO COMPLETE
            slides_result = generate_slides_gamma(self.pitch_info)
            
            if slides_result["type"] == "slides_generated":
                # ACCUMULATE: Add final tool step
                all_tool_steps.append({
                    "type": "tool_result",
                    "name": "Gamma AI Slides Generator",
                    "message": "✅ Pitch deck generated successfully!",
                    "status": "completed",
                    "data": {
                        "slides_url": slides_result["slides_url"],
                        "business_info": slides_result["business_info"]
                    },
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                yield {
                    "type": "tool_result",
                    "tool_name": "Gamma AI Slides Generator",
                    "message": "✅ Pitch deck generated successfully!",
                    "status": "slides_generated"
                }
                
                await asyncio.sleep(0.3)
                
                # Final response with slides
                self.last_generated_slides = slides_result["slides_url"]
                
                yield {
                    "type": "slides_generated",
                    "message": slides_result["message"],
                    "slides_url": slides_result["slides_url"],
                    "business_info": slides_result["business_info"],
                    "status": "complete"
                }
                
                # SEND COMPLETE EVENT WITH ALL ACCUMULATED DATA
                yield {
                    "type": "complete",
                    "status": "complete",
                    "message": slides_result["message"],
                    "final_data": {
                        "slides_url": slides_result["slides_url"],
                        "business_info": slides_result["business_info"],
                        "search_results": formatted_results,
                        "search_keywords": search_keywords,
                        "inspiration_images": inspiration_images,
                        "tool_steps": all_tool_steps,
                        "thinking_process": thinking_process_data
                    }
                }
                
                try:
                    final_message_data = {
                        "sender": "agent",
                        "text": slides_result["message"],
                        "toolSteps": all_tool_steps,
                        "thinkingProcess": thinking_process_data,
                        "searchResults": {
                            "keywords": search_keywords,
                            "results": formatted_results
                        },
                        "inspirationImages": inspiration_images,
                        "slidesUrl": slides_result["slides_url"],
                        "isPitchDeck": True,
                        "status": "complete"
                    }
                    save_result = self.save_rich_message(self.conversation_id, self.user_id, final_message_data)
                    if save_result["type"] == "success":
                        print(f"[DEBUG] Final message saved successfully to DB for conversation {self.conversation_id}")
                    else:
                        print(f"[DEBUG] Failed to save final message: {save_result['message']}")
                except Exception as save_error:
                    print(f"[DEBUG] Error saving final message: {str(save_error)}")
                
            else:
                yield {
                    "type": "error",
                    "message": slides_result["message"],
                    "status": "generation_failed"
                }
                
                # SEND COMPLETE SIGNAL EVEN FOR ERRORS
                yield {
                    "type": "complete",
                    "status": "error",
                    "message": slides_result["message"],
                    "final_data": {
                        "tool_steps": all_tool_steps,
                        "thinking_process": thinking_process_data
                    }
                }
                
                try:
                    error_message_data = {
                        "sender": "agent",
                        "text": slides_result["message"],
                        "toolSteps": all_tool_steps,
                        "thinkingProcess": thinking_process_data,
                        "status": "error"
                    }
                    save_result = self.save_rich_message(self.conversation_id, self.user_id, error_message_data)
                    if save_result["type"] == "success":
                        print(f"[DEBUG] Error message saved successfully to DB for conversation {self.conversation_id}")
                    else:
                        print(f"[DEBUG] Failed to save error message: {save_result['message']}")
                except Exception as save_error:
                    print(f"[DEBUG] Error saving error message: {str(save_error)}")
                    
        except Exception as e:
            yield {
                "type": "error",
                "message": f"Pitch deck generation failed: {str(e)}",
                "status": "error"
            }
            
            # SEND COMPLETE SIGNAL ON EXCEPTION
            yield {
                "type": "complete",
                "status": "error",
                "message": f"Pitch deck generation failed: {str(e)}",
                "final_data": {
                    "tool_steps": all_tool_steps if 'all_tool_steps' in locals() else [],
                    "thinking_process": thinking_process_data if 'thinking_process_data' in locals() else {}
                }
            }

    async def get_real_model_thinking(self, query: str) -> dict:
        """Get REAL model thinking using Groq reasoning model"""
        
        thinking_prompt = f"""
        <thinking>
        The user is asking me: "{query}"
        
        Let me think step by step about this pitch deck request:
        
        1. What exactly is the user asking for?
        2. What type of pitch deck do they need (investor, client, product launch)?
        3. What information will I need to create an effective pitch deck?
        4. What's my strategy for helping them create a compelling presentation?
        5. How can I make this process smooth and efficient?
        
        I need to analyze this carefully to provide the best possible service.
        </thinking>
        
        Analyze this user request for pitch deck creation. Show your complete reasoning process.
        
        USER REQUEST: "{query}"
        
        Think through this step-by-step and show your reasoning.
        """
        
        try:
            response = self.groq_client.chat.completions.create(
                model=self.reasoning_model,
                messages=[
                    {"role": "system", "content": "You are a professional pitch deck expert. Think through client requests step-by-step, showing your complete reasoning process in <thinking> tags."},
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
                "analysis": "Analyzing the user's pitch deck request",
                "plan": "Developing strategy to help achieve their goals"
            }
            
        except Exception as e:
            print(f"Real thinking generation error: {e}")
            return {
                "thinking": f"I'm analyzing your request: {query}. Let me think about what type of pitch deck you need and what information I'll require to create something compelling for your presentation.",
                "reasoning": "Processing user request for pitch deck assistance",
                "analysis": "User wants pitch deck creation help",
                "plan": "Gather business information and create the requested presentation"
            }

    async def get_real_extraction_thinking(self, query: str) -> dict:
        """Show REAL thinking process for pitch deck information extraction using reasoning model"""
        
        extraction_prompt = f"""
        <thinking>
        I need to extract pitch deck information from this request: "{query}"
        
        Let me think carefully:
        1. What business/company information can I identify directly from this text?
        2. What clues about presentation type, audience, or industry are present?
        3. What's missing that I'll need to ask for?
        4. What can I intelligently infer from context?
        5. How should I prioritize the information I gather?
        
        I need to be thorough but efficient in my extraction process.
        </thinking>
        
        Extract pitch deck information from this request, showing your reasoning process.
        
        REQUEST: "{query}"
        
        Think through your extraction strategy step-by-step.
        """
        
        try:
            response = self.groq_client.chat.completions.create(
                model=self.reasoning_model,
                messages=[
                    {"role": "system", "content": "Analyze text for pitch deck information extraction. Show your complete reasoning process in <thinking> tags."},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.3,
                max_tokens=600
            )
            
            thinking_text = response.choices[0].message.content.strip()
            
            # Extract thinking and response
            thinking_match = re.search(r'<thinking>(.*?)</thinking>', thinking_text, re.DOTALL)
            thinking_content = thinking_match.group(1).strip() if thinking_match else thinking_text
            
            response_match = re.search(r'</thinking>\s*(.*)', thinking_text, re.DOTALL)
            process_content = response_match.group(1).strip() if response_match else thinking_text
            
            return {
                "thinking": thinking_content,
                "process": process_content,
                "findings": "Analyzing pitch deck elements in the request"
            }
                
        except Exception as e:
            return {
                "thinking": f"I'm examining this request: {query}. I need to look for business name, presentation type, target audience, and any specific requirements mentioned.",
                "process": f"Extracting information from: {query}",
                "findings": "Looking for business name, presentation type, and requirements"
            }

    async def get_real_slide_thinking(self, pitch_info: dict, user_context: str) -> dict:
        """Show REAL strategic thinking process for slide structure and flow using reasoning model"""
        
        slide_prompt = f"""
        <thinking>
        I'm about to create a pitch deck with this information:
        Business Info: {pitch_info}
        User Context: "{user_context}"
        
        Let me think through the slide strategy:
        1. What slide structure would work best for this business?
        2. How should I order the slides to create a compelling narrative?
        3. Which key slides must be included for this type of pitch?
        4. What storytelling approach will be most effective?
        5. How can I ensure this deck achieves the user's objectives?
        
        This is the strategic planning phase for the presentation.
        </thinking>
        
        Plan the slide structure and presentation strategy for this pitch deck.
        
        BUSINESS INFO: {pitch_info}
        USER CONTEXT: "{user_context}"
        
        Show your strategic reasoning step-by-step.
        """
        
        try:
            response = self.groq_client.chat.completions.create(
                model=self.reasoning_model,
                messages=[
                    {"role": "system", "content": "Think through the pitch deck strategy like a professional presentation designer. Show your reasoning in <thinking> tags."},
                    {"role": "user", "content": slide_prompt}
                ],
                temperature=0.4,
                max_tokens=800
            )
            
            thinking_text = response.choices[0].message.content.strip()
            
            # Extract thinking and response
            thinking_match = re.search(r'<thinking>(.*?)</thinking>', thinking_text, re.DOTALL)
            thinking_content = thinking_match.group(1).strip() if thinking_match else thinking_text
            
            response_match = re.search(r'</thinking>\s*(.*)', thinking_text, re.DOTALL)
            strategy_content = response_match.group(1).strip() if response_match else thinking_text
            
            return {
                "thinking": thinking_content,
                "slide_strategy": strategy_content,
                "key_slides": "Planning essential slides for maximum impact",
                "narrative_flow": "Creating logical progression and compelling story"
            }
                
        except Exception as e:
            return {
                "thinking": "I'm considering how to structure this pitch deck for maximum impact. I need to balance comprehensive information with engaging presentation and ensure the narrative flow serves the business objectives.",
                "slide_strategy": "Developing optimal slide structure and flow",
                "key_slides": "Identifying essential slides based on pitch type",
                "narrative_flow": "Planning compelling narrative progression"
            }

    async def stream_conversation_response(self, query: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream regular conversation responses"""
        
        try:
            if self.conversation_id and self.user_id:
                MongoDB.save_message(
                    conversation_id=self.conversation_id,
                    user_id=self.user_id,
                    sender='user',
                    text=query
                )
                print(f"[DEBUG] User message saved to MongoDB: {query}")
        
            # Store user query in Pinecone
            store_in_pinecone(
                agent_type="pitch-deck", 
                role="user", 
                text=query,
                user_id=self.user_id,
                conversation_id=self.conversation_id
            )
            print(f"[DEBUG] User message stored in Pinecone")
            yield {
                "type": "status",
                "message": "💭 Preparing response...",
                "status": "thinking"
            }
            
            await asyncio.sleep(0.5)
            
            # Use the regular agent response
            response = self.agent.run(query)
            
            # Stream the response word by word for effect
            words = response.split()
            current_text = ""
            
            for i, word in enumerate(words):
                current_text += word + " "
                
                yield {
                    "type": "message_chunk",
                    "text": current_text.strip(),
                    "is_complete": i == len(words) - 1,
                    "status": "streaming"
                }
                
                await asyncio.sleep(0.05)  # Small delay between words
            
            # Final complete message
            yield {
                "type": "message",
                "text": response,
                "status": "complete"
            }
            yield {
                "type": "complete",
                "status": "complete"
            }
            
            # Save to MongoDB
            if self.conversation_id and self.user_id:
                MongoDB.save_message(
                    conversation_id=self.conversation_id,
                    user_id=self.user_id,
                    sender='agent',
                    text=response,
                    agent=self.agent_name
                )
                
        except Exception as e:
            yield {
                "type": "error",
                "message": f"Response generation failed: {str(e)}",
                "status": "error"
            }
            yield {
                "type": "complete",
                "status": "error"
            }

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

    def ask_comprehensive_pitch_questions(self) -> str:
        """Generate comprehensive questions for pitch deck creation"""
        questions_prompt = """
        You are a professional pitch deck creator about to help a client create a compelling presentation.
        Generate comprehensive questions to gather all necessary information for a pitch deck in ONE message.
        
        REQUIREMENTS:
        1. Ask for ALL essential information in one message (don't make them wait)
        2. Use emojis and clear formatting to make it easy to read
        3. Provide examples to help them respond
        4. Show enthusiasm about the project
        5. Make it feel conversational and friendly
        
        ESSENTIAL INFORMATION TO ASK FOR:
        - Business/Company name
        - Presentation purpose (investor pitch, client proposal, product launch)
        - Target audience (investors, clients, partners)
        - Industry/market
        - Problem being solved
        - Solution offered
        - Business model
        - Market size/opportunity
        - Funding needs (if applicable)
        
        Generate natural, comprehensive questions that cover all these points for a pitch deck.
        """
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Generate comprehensive, friendly questions for gathering pitch deck requirements. Be enthusiastic and helpful."},
                    {"role": "user", "content": questions_prompt}
                ],
                temperature=0.8,
                max_tokens=500
            )
            
            dynamic_questions = response.choices[0].message.content.strip()
            print(f"[DEBUG] Generated comprehensive pitch deck questions")
            
            return dynamic_questions
            
        except Exception as e:
            print(f"[DEBUG] Questions generation error: {e}")
            
            # Fallback questions
            return """
            I'm excited to help create your pitch deck! To make it perfect, I'll need some key information:

            🏢 **Business Name:** What's your company/startup called?
            
            🎯 **Presentation Purpose:** Is this for investors, clients, or product launch?
            
            👥 **Target Audience:** Who will be viewing this presentation?
            
            🔍 **Industry:** What industry are you in?
            
            ❓ **Problem:** What problem does your business solve?
            
            💡 **Solution:** How does your business solve this problem?
            
            💰 **Business Model:** How does your business make money?
            
            📈 **Market Size:** What's your target market and its size?
            
            💸 **Funding Ask:** Are you seeking investment? How much?
            
            Example: "TechSolve, investor pitch for Series A, targeting VCs in the healthcare tech space. We solve medical scheduling problems with our AI platform. Subscription-based revenue model in a $50B market. Seeking $2M in funding."
            
            Feel free to share as much detail as you'd like!
            """

    def convert_to_keywords(self, query: str) -> str:
        """Convert user query to search keywords using GPT"""
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Convert the user query into a keyword optimized for search engine query to find pitch deck examples/help/guidance. Return single keyword only."},
                    {"role": "user", "content": f"User wants: {query}. Extract pitch deck-related search keyword."}
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

    def search_with_keywords(self, keywords: str):
        """Search web content using SearchAPI.io"""
        params = {
            "engine": "google",
            "q": keywords + " pitch deck examples",
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

    def format_search_results(self, results):
        """Format web search results for display in frontend Sources Modal"""
        articles = []
        for item in results.get("organic_results", []):
            articles.append({
                "title": item.get("title"),
                "link": item.get("link"), 
                "source": item.get("source"),
                "snippet": item.get("snippet")
            })
        return articles[:8]


def get_pitch_deck_agent(user_id: str = None, conversation_id: str = None):
    return PitchDeckAgent(user_id, conversation_id)