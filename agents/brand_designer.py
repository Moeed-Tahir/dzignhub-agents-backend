import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool, AgentType
from core.config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENV, GROQ_API_KEY, SEARCHAPI_KEY
from core.database import MongoDB
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime
import asyncio
from typing import AsyncGenerator, Dict, Any
from groq import Groq
import re
import json
import requests
# ---------------------------
# Pinecone Setup (same as before)
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
# Helper Functions (same as before)
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



def search_conversations_by_query(query: str, user_id: str, agent_type: str = "brand-designer", top_k: int = 10):
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


def generate_logo_dalle(info: dict):
    """Generate logo with DALL-E and return proper format for frontend"""
    
    # Validate required information
    brand_name = info.get('brand_name', 'Brand')
    logo_type = info.get('logo_type', 'modern professional')
    target_audience = info.get('target_audience', 'general audience')
    color_palette = info.get('color_palette', 'professional colors')
    
    # Create a detailed and specific prompt for DALL-E
    prompt = f"""Create a {logo_type} logo design for "{brand_name}". 
Target audience: {target_audience}. 
Color scheme: {color_palette}. 
Style: Clean, professional, modern, and memorable. 
High quality vector-style design suitable for business branding. 
Simple, scalable design that works well at different sizes. 
No text overlay or watermarks. 
Professional business logo format."""
    
    try:
        print(f"[DEBUG] Generating logo with DALL-E...")
        print(f"[DEBUG] Brand: {brand_name}")
        print(f"[DEBUG] Style: {logo_type}")
        print(f"[DEBUG] Audience: {target_audience}")
        print(f"[DEBUG] Colors: {color_palette}")
        print(f"[DEBUG] Prompt: {prompt}")
        
        # Generate image using DALL-E 3
        result = openai_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        
        # Extract the image URL
        image_url = result.data[0].url
        print(f"[DEBUG] Successfully generated logo!")
        print(f"[DEBUG] Image URL: {image_url}")
        
        # Create success message
        success_message = f"🎉 **Your {brand_name} logo is ready!**\n\nI've created a {logo_type} design that perfectly captures your brand identity for {target_audience}. The design uses {color_palette} to create a professional and memorable look."
        
        # Return successful result
        return {
            "type": "logo_generated",
            "message": success_message,
            "image_url": image_url,
            "brand_info": {
                "brand_name": brand_name,
                "logo_type": logo_type,
                "target_audience": target_audience,
                "color_palette": color_palette
            }
        }
        
    except Exception as e:
        print(f"[DEBUG] DALL-E generation error: {str(e)}")
        
        # Return error result
        return {
            "type": "error",
            "message": f"I encountered an issue generating your logo: {str(e)}. Let me try again with different specifications, or you can adjust your requirements.",
            "image_url": None,
            "brand_info": info
        }
# ---------------------------
# Brand Designer Agent with MongoDB
# ---------------------------
class BrandDesignerAgent:
    def __init__(self, user_id: str = None, conversation_id: str = None):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.agent_name = "brand-designer"
        self.last_generated_image = None
        self.groq_client = groq_client
        self.reasoning_model = "deepseek-r1-distill-llama-70b"
        self.show_thinking = True  # Enable thinking display
        self.searchapi_key = SEARCHAPI_KEY
        self.search_base_url = "https://www.searchapi.io/api/v1/search"
        print(f"[DEBUG] Initialized with reasoning model: {self.reasoning_model}")
        

         # Enhanced design info for multiple asset types
        self.design_info = {
            "brand_name": None,
            "logo_type": None,
            "target_audience": None,
            "color_palette": None,
            "brand_personality": None,  # New: professional, creative, playful, etc.
            "industry": None,           # New: tech, healthcare, food, etc.
            "preferred_fonts": None,    # New: modern, classic, bold, etc.
        }


        # Add the system prompt as an attribute
        self.system_prompt = """You are Zara, a professional brand designer assistant who creates comprehensive visual brand assets and provides design consultation.

Your capabilities:
🎨 **Visual Brand Assets:**
- Logo design (text-based, icon-based, combination, mascot)
- Social media graphics (Instagram posts, LinkedIn covers, Facebook banners)
- Marketing materials (posters, flyers, thumbnails)
- Business materials (business cards, letterheads, presentations)
- Web graphics (banners, hero images, backgrounds)

🎯 **Design Consultation:**
- Color palette recommendations and psychology
- Typography guidance and font selection
- Brand identity strategy
- Asset optimization for different platforms

When users request any visual asset, immediately use the Generate_Brand_Asset tool.

Key guidelines:
- Be creative and professional
- Consider platform-specific requirements
- Maintain brand consistency across all assets
- Ask clarifying questions when needed

Always prioritize using the tool for any visual design requests."""

        self.llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)  # Updated model
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.design_info = {
            "brand_name": None,
            "logo_type": None,
            "target_audience": None,
            "color_palette": None
        }
        # Load conversation history if conversation_id exists
        if self.conversation_id and self.user_id:
            self.load_conversation_history()
            self.load_brand_design()

        # Create the logo generation tool with smart info collection
        tools = [
            Tool(
                name="Generate_Brand_Asset",
                func=self.smart_asset_generator,
                description="""Use this tool IMMEDIATELY when the user requests ANY visual brand asset including:
                
                LOGOS: text-based, icon-based, combination, mascot logos
                SOCIAL MEDIA: Instagram posts, LinkedIn covers, Facebook banners, Twitter headers, YouTube thumbnails
                MARKETING: posters, flyers, brochures, advertisements, banners
                BUSINESS: business cards, letterheads, presentations, email signatures
                WEB: hero images, website banners, backgrounds, covers
                
                This tool handles information collection AND actual asset generation for ALL visual design requests.
                Always use this tool for any design generation requests."""
            )
        ]

        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            agent_kwargs={
    'system_message': """You are Zara, a professional brand designer assistant who helps users create comprehensive visual brand assets.

Your capabilities:
- Logo design and generation using DALL-E
- Social media graphics (LinkedIn covers, Instagram posts, Facebook banners) 
- Marketing materials (posters, flyers, brochures)
- Business materials (business cards, letterheads)

When users mention wanting ANY visual asset, immediately use the Generate_Brand_Asset tool.

CRITICAL INSTRUCTION: When the tool returns a response that starts with "ASSET_GENERATED|", you MUST return that EXACT response without any modifications, additions, or formatting changes. Do not convert it to markdown, do not add explanations, just return the exact string as-is.

Examples:
- If tool returns: "ASSET_GENERATED|https://image-url|message"
- You return: "ASSET_GENERATED|https://image-url|message" (EXACTLY)

For all other responses, be conversational and friendly.

Key guidelines:
- Always use the tool for ANY design-related requests (logos, LinkedIn covers, posters, etc.)
- NEVER modify ASSET_GENERATED responses
- Trust the tool to handle information collection and generation
- Focus on being helpful and creative for non-design conversations

Always prioritize using the tool over giving generic advice."""
}
    )

    def extract_brand_info_from_conversation(self, messages):
        """Use GPT to intelligently extract brand information from conversation"""
        
        # Build FULL conversation text - ALL messages, not just recent
        conversation_text = ""
        for msg in messages:
            role = "User" if msg['sender'] == 'user' else "Assistant"
            conversation_text += f"{role}: {msg['text']}\n"
        
        if not conversation_text.strip():
            print("[DEBUG] No conversation history to extract from")
            return
        
        print(f"[DEBUG] Extracting from FULL conversation: {conversation_text}")
        
        # Enhanced extraction with brand switching detection
        extraction_prompt = f"""
        Analyze this COMPLETE conversation and extract brand information. 
        CRITICAL: Detect when user mentions a DIFFERENT brand name than previously discussed.
        
        Conversation:
        {conversation_text}

        BRAND NAME CHANGE DETECTION RULES:
        1. If user says "but my brand name is X" - they are CORRECTING the brand name
        2. If user says "my brand is X" after talking about different brand - BRAND CHANGE
        3. If user mentions "AllMyAi" after discussing "FashFoo" - BRAND CHANGE
        4. Always use the MOST RECENT brand name mentioned
        5. When brand changes, RESET all other fields (logo_type, colors, etc.)

        BRAND SWITCHING PATTERNS:
        - "but my brand name is [Name]" = CORRECTION
        - "my brand is [Name]" = POSSIBLE SWITCH  
        - "for [BrandName]" = CURRENT BRAND
        - "[BrandName] logo" = CURRENT BRAND

        Extract the MOST RECENT brand information:
        - brand_name: The latest brand name mentioned (prioritize corrections)
        - asset_type: What they want to create (logo, instagram_post, etc.)
        - is_new_brand: true if brand name is different from what was discussed earlier
        - Other fields: Only keep if they're for the CURRENT brand

        Return ONLY valid JSON:
        {{"brand_name": "most recent brand name or null", "asset_type": "logo/instagram_post/etc or null", "is_new_brand": true/false, "logo_type": null, "target_audience": null, "color_palette": null}}
        """
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Detect brand name changes in conversations. When user says 'but my brand name is X', this is a brand correction. Return ONLY valid JSON."},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.1,
                max_tokens=300
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
            
            # Check if this is a brand change
            current_brand = self.design_info.get('brand_name')
            new_brand = extracted_info.get('brand_name')
            
            # BRAND CHANGE DETECTION
            if new_brand and current_brand and new_brand.lower() != current_brand.lower():
                print(f"[DEBUG] BRAND CHANGE DETECTED: {current_brand} → {new_brand}")
                extracted_info['is_new_brand'] = True
            
            # If brand changed or explicitly marked as new brand, RESET everything
            if extracted_info.get("is_new_brand"):
                print("[DEBUG] Resetting design_info for new brand")
                self.design_info = {
                    "brand_name": None,
                    "logo_type": None,
                    "target_audience": None,
                    "color_palette": None,
                    "brand_personality": None,
                    "industry": None,
                    "preferred_fonts": None,
                }
            
            # Update with new brand information
            for key, value in extracted_info.items():
                if key in self.design_info and value and value.lower() not in ["null", "", "none"]:
                    self.design_info[key] = value
                    print(f"[DEBUG] Updated {key}: {value}")
            
            # Store asset type
            if extracted_info.get("asset_type"):
                self.detected_asset_type = extracted_info["asset_type"]
                print(f"[DEBUG] Detected asset type: {self.detected_asset_type}")
            
            print(f"[DEBUG] Final design_info: {self.design_info}")
            
        except Exception as e:
            print(f"[DEBUG] Extraction error: {e}")
            
            # MANUAL FALLBACK - Look for brand names in ALL messages
            print("[DEBUG] Using manual brand name search")
            for msg in reversed(messages):  # Start from most recent
                text = msg['text'].lower()
                
                import re
                patterns = [
                    r"but my brand name is (\w+)",
                    r"my brand name is (\w+)",  
                    r"my brand is (\w+)",
                    r"brand (\w+)",
                    r"(\w+) logo"
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, text)
                    if match:
                        found_brand = match.group(1).title()  # Capitalize
                        current_brand = self.design_info.get('brand_name')
                        
                        # Skip common words
                        if found_brand.lower() in ['the', 'my', 'our', 'logo', 'design']:
                            continue
                        
                        # If different from current, reset
                        if current_brand and found_brand != current_brand:
                            print(f"[DEBUG] Manual brand change: {current_brand} → {found_brand}")
                            self.design_info = {
                                "brand_name": found_brand,
                                "logo_type": None,
                                "target_audience": None,
                                "color_palette": None,
                                "brand_personality": None,
                                "industry": None,
                                "preferred_fonts": None,
                            }
                            return
                        
                        self.design_info['brand_name'] = found_brand
                        print(f"[DEBUG] Manual extraction: {found_brand}")
                        return


    def intelligent_auto_complete(self, provided_info: dict, asset_type: str = "logo"):
        """Enhanced auto-completion with asset-type awareness"""
        
        known_info = ""
        missing_fields = []
        for key, value in provided_info.items():
            if value:
                known_info += f"{key}: {value}\n"
            else:
                missing_fields.append(key)
        
        if not known_info.strip():
            return provided_info
        
        # FIXED: No f-string with JSON braces
        brand_name = provided_info.get('brand_name', 'Modern Brand')
        
        completion_prompt = f"""
        Based on the following brand information, intelligently suggest appropriate values for the missing fields.
        The user wants to create a {asset_type.replace('_', ' ')}.

        Known information:
        {known_info}

        Please complete this JSON with intelligent defaults for the missing fields: {missing_fields}

        Guidelines based on asset type "{asset_type}":
        - For social media assets: Consider platform-specific audience behavior
        - For business materials: Prioritize professionalism and clarity
        - For marketing materials: Focus on eye-catching and memorable elements
        - For logos: Consider scalability and versatility
        
        Industry-specific suggestions:
        - Tech companies: Modern, minimalist, tech-friendly colors (blues, grays, greens)
        - Food/Restaurant: Warm, appetizing colors (reds, oranges, yellows)
        - Healthcare: Trust-building colors (blues, whites, greens)
        - Finance: Professional, trustworthy (navy, gray, gold)
        - Creative/Design: Vibrant, artistic (varied creative palette)

        Return ONLY a complete JSON object with all fields.
        Format: """ + '{"brand_name": "' + brand_name + '", "logo_type": "suggest based on brand and asset type", "target_audience": "suggest based on brand and asset type", "color_palette": "suggest based on brand and asset type", "brand_personality": "suggest based on brand context", "industry": "suggest based on brand name", "preferred_fonts": "suggest based on brand personality"}'
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert brand consultant with deep knowledge of asset-specific design requirements. Always return valid JSON."},
                    {"role": "user", "content": completion_prompt}
                ],
                temperature=0.3,
                max_tokens=400
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

     # ✅ ADD: Search functionality methods
    def search_with_keywords(self, keywords: str):
        """Search web content using SearchAPI.io with better error handling"""
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
                    {"role": "system", "content": "Convert the user query into a keyword optimized for search engine query to find inspirations/help/guidance for design. Return single keyword only."},
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
        """Search for design inspiration images from Behance and Dribbble"""
        print(f"[DEBUG] Starting image search for: '{query}' with {num_results} results")
        
        # Search queries for each platform
        behance_query = f"site:behance.net {query}"
        dribbble_query = f"site:dribbble.com {query}"
        
        print(f"[DEBUG] Behance query: {behance_query}")
        print(f"[DEBUG] Dribbble query: {dribbble_query}")
        
        all_images = []
        
        # Search Behance - Get exactly 5 images
        try:
            params = {
                "engine": "google_images",
                "q": behance_query,
                "api_key": self.searchapi_key,
                "num": 5  # ✅ Request exactly 5 images
            }
            
            
            
            response = requests.get(self.search_base_url, params=params, timeout=10)
            
            
            
            if response.status_code == 200:
                behance_data = response.json()
                print(f"[DEBUG] Behance response keys: {behance_data.keys()}")
                
                # ✅ Get images from correct field
                images_list = behance_data.get("images", [])
                print(f"[DEBUG] Behance total images available: {len(images_list)}")
                
                # ✅ Process exactly 5 images from Behance
                for i, item in enumerate(images_list[:5]):  # Take only first 5
                    print(f"[DEBUG] Processing Behance image {i+1}: {item.get('title', 'No title')[:50]}...")
                    
                    # ✅ Extract image data with correct structure
                    original_data = item.get("original", {})
                    source_data = item.get("source", {})
                    
                    image_data = {
                        "title": item.get("title", "Behance Design")[:100],  # Limit title length
                        "original": original_data.get("link") if isinstance(original_data, dict) else original_data,
                        "thumbnail": item.get("thumbnail"),
                        "link": source_data.get("link") if isinstance(source_data, dict) else None,
                        "source": "Behance",
                        "position": item.get("position", i+1)
                    }
                    
                    # ✅ Only add if we have valid image URLs
                    if image_data["original"] and image_data["thumbnail"]:
                        all_images.append(image_data)
                        print(f"[DEBUG] ✅ Added Behance image {len(all_images)}")
                    else:
                        print(f"[DEBUG] ❌ Skipped Behance image {i+1} - missing URLs")
                        
            else:
                print(f"[DEBUG] Behance API error: {response.status_code}")
                print(f"[DEBUG] Behance error response: {response.text}")
                
        except Exception as e:
            print(f"[DEBUG] Behance search error: {e}")
            import traceback
            print(f"[DEBUG] Behance traceback: {traceback.format_exc()}")

        # Search Dribbble - Get exactly 5 images
        try:
            params = {
                "engine": "google_images", 
                "q": dribbble_query,
                "api_key": self.searchapi_key,
                "num": 5  # ✅ Request exactly 5 images
            }
            
            print(f"[DEBUG] Dribbble request params: {params}")
            
            response = requests.get(self.search_base_url, params=params, timeout=10)
            
            print(f"[DEBUG] Dribbble response status: {response.status_code}")
            
            if response.status_code == 200:
                dribbble_data = response.json()
                print(f"[DEBUG] Dribbble response keys: {dribbble_data.keys()}")
                
                # ✅ Get images from correct field
                images_list = dribbble_data.get("images", [])
                print(f"[DEBUG] Dribbble total images available: {len(images_list)}")
                
                # ✅ Process exactly 5 images from Dribbble
                for i, item in enumerate(images_list[:5]):  # Take only first 5
                    print(f"[DEBUG] Processing Dribbble image {i+1}: {item.get('title', 'No title')[:50]}...")
                    
                    # ✅ Extract image data with correct structure
                    original_data = item.get("original", {})
                    source_data = item.get("source", {})
                    
                    image_data = {
                        "title": item.get("title", "Dribbble Design")[:100],  # Limit title length
                        "original": original_data.get("link") if isinstance(original_data, dict) else original_data,
                        "thumbnail": item.get("thumbnail"),
                        "link": source_data.get("link") if isinstance(source_data, dict) else None,
                        "source": "Dribbble",
                        "position": item.get("position", i+1)
                    }
                    
                    # ✅ Only add if we have valid image URLs
                    if image_data["original"] and image_data["thumbnail"]:
                        all_images.append(image_data)
                        print(f"[DEBUG] ✅ Added Dribbble image {len(all_images)}")
                    else:
                        print(f"[DEBUG] ❌ Skipped Dribbble image {i+1} - missing URLs")
                        
            else:
                print(f"[DEBUG] Dribbble API error: {response.status_code}")
                print(f"[DEBUG] Dribbble error response: {response.text}")
                
        except Exception as e:
            print(f"[DEBUG] Dribbble search error: {e}")
            import traceback
            print(f"[DEBUG] Dribbble traceback: {traceback.format_exc()}")

        print(f"[DEBUG] Final result: {len(all_images)} design inspiration images collected")
        print(f"[DEBUG] Images breakdown:")
        for i, img in enumerate(all_images):
            print(f"[DEBUG] {i+1}. {img['source']}: {img['title'][:50]}...")
        
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
        return articles[:8]  # Increase to 8 results for better sources


    
    def extract_from_current_input(self, user_input: str):
        """Extract brand info from current input with better brand switching detection"""
        
        current_info_text = ""
        if any(v for v in self.design_info.values()):
            current_info_text = f"Currently working on:\n"
            for key, value in self.design_info.items():
                if value:
                    current_info_text += f"- {key}: {value}\n"
            current_info_text += "\n"
        
        # SIMPLIFIED extraction prompt that FORCES JSON output
        extraction_prompt = f"""
        {current_info_text}User input: "{user_input}"
        
        Extract information and return ONLY valid JSON, no explanations:
        
        Asset type mapping:
        - "instagram poster/post" → "instagram_post"
        - "linkedin poster/cover" → "linkedin_cover"
        - "facebook poster/cover" → "facebook_cover"
        - "generate asset" → "logo" (default)
        
        CRITICAL: Return ONLY the JSON object, no other text.
        """
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "CRITICAL: Return ONLY valid JSON with no explanations, no text before or after. Just the JSON object."},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.1,
                max_tokens=150  # Shorter to force concise responses
            )
            
            extracted_text = response.choices[0].message.content.strip()
            print(f"[DEBUG] Current input extraction: {extracted_text}")
            
            # More aggressive cleanup
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
            
            # Rest of the logic stays the same...
            # If it's a new brand, reset design_info
            if extracted_info.get("is_new_brand"):
                print("[DEBUG] New brand detected from current input, resetting design_info")
                self.design_info = {
                    "brand_name": None,
                    "logo_type": None,
                    "target_audience": None,
                    "color_palette": None,
                    "brand_personality": None,
                    "industry": None,
                    "preferred_fonts": None,
                }
            
            # Update design_info with new extracted information
            for key, value in extracted_info.items():
                if key in self.design_info and value and value.lower() not in ["null", "", "none"]:
                    self.design_info[key] = value
                    print(f"[DEBUG] Updated from current input - {key}: {value}")
            
            # Store the asset type separately if detected
            if extracted_info.get("asset_type"):
                self.detected_asset_type = extracted_info["asset_type"]
                print(f"[DEBUG] Detected asset type from current input: {self.detected_asset_type}")
            
            print(f"[DEBUG] Final design_info after current input: {self.design_info}")
                    
        except Exception as e:
            print(f"[DEBUG] Current input extraction error: {e}")
            if 'extracted_text' in locals():
                print(f"[DEBUG] Raw extraction text: {extracted_text}")
            
            # SIMPLE FALLBACK - just extract asset type from keywords
            print("[DEBUG] Using simple keyword fallback")
            user_lower = user_input.lower()
            
            if "instagram" in user_lower and ("post" in user_lower or "poster" in user_lower):
                self.detected_asset_type = "instagram_post"
            elif "linkedin" in user_lower and ("cover" in user_lower or "poster" in user_lower):
                self.detected_asset_type = "linkedin_cover"
            elif "facebook" in user_lower and ("cover" in user_lower or "poster" in user_lower):
                self.detected_asset_type = "facebook_cover"
            elif "logo" in user_lower:
                self.detected_asset_type = "logo"
            elif "poster" in user_lower:
                self.detected_asset_type = "poster"
            elif any(word in user_lower for word in ["generate", "create", "make"]):
                self.detected_asset_type = "logo"  # Default
            
            print(f"[DEBUG] Fallback detected asset type: {self.detected_asset_type}")


    def load_brand_design(self):
        """Load brand design from User.brandDesign field"""
        if not self.user_id:
            return
            
        try:
            brand_design = MongoDB.get_user_brand_design(self.user_id)
            
            # Update design_info with saved data
            for key, value in brand_design.items():
                if key in self.design_info and value:
                    self.design_info[key] = value
                    print(f"[DEBUG] Loaded {key}: {value}")
            
            print(f"[DEBUG] Loaded brand design: {self.design_info}")
                
        except Exception as e:
            print(f"[DEBUG] Error loading brand design: {e}")

    def save_brand_design(self):
        """Save current design_info to User.brandDesign field"""
        if not self.user_id:
            return
            
        try:
            # Filter out None values and add timestamp
            brand_design_data = {k: v for k, v in self.design_info.items() if v is not None}
            brand_design_data["lastUpdated"] = datetime.utcnow().isoformat()
            
            success = MongoDB.update_user_brand_design(self.user_id, brand_design_data)
            
            if success:
                print(f"[DEBUG] Saved brand design: {brand_design_data}")
            else:
                print("[DEBUG] Failed to save brand design")
                
        except Exception as e:
            print(f"[DEBUG] Error saving brand design: {e}")

    def detect_generation_intent(self, user_input: str) -> bool:
        """Enhanced generation intent detection for all asset types"""
        
        intent_prompt = f"""
        Analyze this user message and determine if they want to generate/create any visual asset NOW.
        
        User message: "{user_input}"
        
        Return ONLY "YES" if they want to generate any visual asset now, or "NO" if they're just providing information.
        
        Examples that mean YES (generate any asset):
        - "create logo" → YES
        - "generate Instagram post" → YES  
        - "make LinkedIn cover" → YES
        - "design poster" → YES
        - "create business card" → YES
        - "generate thumbnail" → YES
        - "make banner" → YES
        - "design flyer" → YES
        - "I need a YouTube thumbnail" → YES
        - "create it now" → YES
        
        Examples that mean NO (just providing info):
        - "my brand name is PsychoDevs" → NO
        - "I want modern colors" → NO
        - "what dimensions work best?" → NO
        
        Answer: """
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Detect if user wants to generate/create ANY visual asset NOW. Always respond with only YES or NO."},
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
            # Enhanced fallback keywords for all asset types
            generation_keywords = [
                "generate", "create", "make", "design", "build",
                "logo", "poster", "banner", "flyer", "thumbnail", 
                "instagram", "linkedin", "facebook", "youtube",
                "business card", "letterhead", "brochure"
            ]
            
            return any(phrase in user_input.lower() for phrase in generation_keywords)

    def smart_asset_generator(self, user_input: str = "") -> str:
        """Smart brand asset generation with better brand switching"""
        
        print(f"[DEBUG] smart_asset_generator called with input: '{user_input}'")
        print(f"[DEBUG] Initial design_info: {self.design_info}")
        
        # ENHANCED IMMEDIATE BRAND NAME CHANGE CHECK
        if user_input:
            user_lower = user_input.lower()
            
            # More comprehensive brand change patterns
            brand_change_patterns = [
                (r"but my brand name is (\w+)", "correction"),
                (r"my brand name is (\w+)", "declaration"),
                (r"my brand is (\w+)", "declaration"),
                (r"brand called (\w+)", "specification"),
                (r"for my brand (\w+)", "specification"),
                (r"(\w+) logo", "implicit_brand"),
            ]
            
            for pattern, intent in brand_change_patterns:
                import re
                match = re.search(pattern, user_lower)
                if match:
                    new_brand_name = match.group(1).title()
                    current_brand = self.design_info.get('brand_name')
                    
                    # Skip common words
                    if new_brand_name.lower() in ['the', 'my', 'our', 'logo', 'design', 'create', 'generate']:
                        continue
                    
                    # Check if this is actually a brand change
                    if current_brand and new_brand_name.lower() != current_brand.lower():
                        print(f"[DEBUG] IMMEDIATE brand change detected ({intent}): {current_brand} → {new_brand_name}")
                        
                        # RESET everything for new brand
                        self.design_info = {
                            "brand_name": new_brand_name,
                            "logo_type": None,
                            "target_audience": None,
                            "color_palette": None,
                            "brand_personality": None,
                            "industry": None,
                            "preferred_fonts": None,
                        }
                        print(f"[DEBUG] Reset design_info for new brand: {self.design_info}")
                        break
                    elif not current_brand:
                        # No current brand, set the new one
                        self.design_info['brand_name'] = new_brand_name
                        print(f"[DEBUG] Set initial brand name: {new_brand_name}")
                        break
        
        
        self.user_context = user_input
        self.detected_asset_type = None
        conversation_detected_type = None
        
        # Process input...
        processed_input = user_input
        if user_input.startswith('{"') and user_input.endswith('}'):
            try:
                import json
                parsed_input = json.loads(user_input)
                processed_input = parsed_input.get("description", "generate asset")
            except:
                pass
        
        # Get ALL conversation messages (not just recent)
        recent_messages = []
        print(f"Getting messages from {self.user_id}. for converastion: {self.conversation_id}.")
        if self.conversation_id and self.user_id:
            messages = MongoDB.get_conversation_messages(self.conversation_id, self.user_id)
            recent_messages = messages  # Use ALL messages
            print(f"[DEBUG] Found {len(recent_messages)} total messages for analysis")
        
        # Extract information from FULL conversation
        if recent_messages:
            self.extract_brand_info_from_conversation(recent_messages)
            conversation_detected_type = getattr(self, 'detected_asset_type', None)
            print(f"[DEBUG] Conversation detected type: {conversation_detected_type}")
        
        # Continue with existing logic for asset generation...
        if processed_input and processed_input.strip():
            self.extract_from_current_input(processed_input)
            current_detected_type = getattr(self, 'detected_asset_type', None)
            print(f"[DEBUG] Current input detected type: {current_detected_type}")
    
        # PRIORITY LOGIC: Use conversation detection if it exists, otherwise use current input
        if conversation_detected_type:
            final_asset_type = conversation_detected_type
            print(f"[DEBUG] Using conversation detected type: {final_asset_type}")
        elif hasattr(self, 'detected_asset_type') and self.detected_asset_type:
            final_asset_type = self.detected_asset_type
            print(f"[DEBUG] Using current input detected type: {final_asset_type}")
        else:
            # Fall back to GPT detection
            asset_info = self.detect_asset_type_and_specs(processed_input)
            final_asset_type = asset_info["type"]
            print(f"[DEBUG] Using GPT fallback detection: {final_asset_type}")
        
        # Handle consultancy requests (no specific brand)
        if final_asset_type == "consultancy":
            return self.provide_brand_consultancy(user_input)
        
        # Get dimensions for the detected type
        type_to_dimensions = {
            "logo": "1024x1024",
            "poster": "1080x1350",
            "instagram_post": "1080x1080",
            "instagram_story": "1080x1920",
            "linkedin_cover": "1584x396",
            "facebook_cover": "1200x630",
            "youtube_thumbnail": "1280x720",
            "business_card": "1050x600"
        }
        dimensions = type_to_dimensions.get(final_asset_type, "1024x1024")
        
        print(f"[DEBUG] Final asset type: {final_asset_type}, dimensions: {dimensions}")
        
        # Continue with the rest of the logic using final_asset_type instead of asset_type
        asset_type = final_asset_type
        
         # COMPREHENSIVE QUESTIONS APPROACH - Ask everything at once
        if not self.design_info.get("brand_name"):
            return self.ask_comprehensive_asset_questions(asset_type)
        
        # Save updated design_info
        self.save_brand_design()
        
        print(f"[DEBUG] Current design_info after extraction: {self.design_info}")
        
        # Check missing information
        missing_info = [k for k, v in self.design_info.items() if not v]
        provided_info = {k: v for k, v in self.design_info.items() if v}
        
        print(f"[DEBUG] Missing info: {missing_info}")
        print(f"[DEBUG] Provided info: {provided_info}")
        
        # Detect generation intent
        wants_generation = self.detect_generation_intent(user_input) if user_input else False
        
        # Generate asset if requested and we have minimum info
        if wants_generation and provided_info.get("brand_name"):
            # Auto-complete missing fields
            if missing_info:
                print("[DEBUG] Auto-completing missing fields for generation...")
                auto_completed = self.intelligent_auto_complete(self.design_info.copy(), asset_type)
                for key, value in auto_completed.items():
                    if not self.design_info.get(key):
                        self.design_info[key] = value
                self.save_brand_design()
            
            print(f"[DEBUG] Generating {asset_type} with dimensions {dimensions}...")
            asset_result = self.generate_brand_asset_dalle(self.design_info, asset_type, dimensions, user_context=user_input)
            
            if asset_result["type"] == "asset_generated":
                self.last_generated_image = asset_result["image_url"]
                return f"""ASSET_GENERATED|{asset_result['image_url']}|{asset_result['message']}"""
            else:
                return asset_result["message"]
        
        # Use natural conversation collection (instead of rigid collect_asset_info)
        return self.collect_asset_info(asset_type, missing_info, provided_info)


    def ask_comprehensive_asset_questions(self, asset_type: str) -> str:
        """Generate dynamic comprehensive questions using GPT based on asset type"""
        asset_display = asset_type.replace('_', ' ')
        
        # Build dynamic prompt for comprehensive questions
        questions_prompt = f"""
        You are a professional brand designer about to create a {asset_display} for a client.
        You need to ask comprehensive questions to gather all necessary information in ONE message.
        
        Generate a friendly, comprehensive question that asks for ALL the key information needed to create a {asset_display}.
        
        REQUIREMENTS:
        1. Ask for ALL essential information in one message (don't make them wait)
        2. Use emojis and clear formatting to make it easy to read
        3. Provide examples to help them respond
        4. Show enthusiasm about the project
        5. Make it feel conversational like ChatGPT would
        6. Include a simple example response format at the end
        
        ESSENTIAL INFORMATION TO ASK FOR:
        - Brand/Company name
        - Industry/Purpose  
        - Style preference
        - Color preferences
        - Target audience
        - {asset_display}-specific requirements
        
        ASSET-SPECIFIC CONSIDERATIONS:
        - For logos: Ask about logo style (text-based, icon, combination)
        - For social media posts: Ask about post purpose/message/content
        - For LinkedIn covers: Ask about professional title/expertise to highlight
        - For business cards: Ask about contact information and role
        - For marketing materials: Ask about key message/call-to-action
        - For Instagram posts: Ask about post type (announcement, product, quote, etc.)
        - For posters: Ask about event/promotion details and key message
        
        Generate a natural, comprehensive question that covers all these points for a {asset_display}.
        """
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": f"Generate comprehensive, friendly questions for gathering {asset_display} design requirements. Be enthusiastic and helpful like ChatGPT."},
                    {"role": "user", "content": questions_prompt}
                ],
                temperature=0.8,  # Higher temperature for more natural, varied responses
                max_tokens=400
            )
            
            dynamic_questions = response.choices[0].message.content.strip()
            print(f"[DEBUG] Generated dynamic comprehensive questions for {asset_type}")
            
            return dynamic_questions
            
        except Exception as e:
            print(f"[DEBUG] Dynamic questions generation error: {e}")
            
            # Intelligent fallback based on asset type
            return self.generate_fallback_questions(asset_type)

    def generate_fallback_questions(self, asset_type: str) -> str:
        """Generate fallback questions when GPT fails"""
        
        asset_display = asset_type.replace('_', ' ')
        
        # Base structure for different asset types
        fallback_templates = {
            "logo": f"""Perfect! I'm excited to create a logo for you. To design something amazing, could you share:

    🏢 **Brand Name:** What's your company/brand called?
    🎯 **Industry:** What kind of business? (tech, restaurant, consulting, etc.)
    🎨 **Style:** What vibe do you want? (modern, playful, professional, creative)
    🌈 **Colors:** Any color preferences or should I surprise you?
    👥 **Audience:** Who are your main customers?

    Example: *"TechFlow, tech startup, modern minimal style, blue colors, young professionals"*""",

            "instagram_post": f"""Great! I'll create an Instagram post for you. Let me know:

    🏢 **Brand:** What's your brand name?
    📱 **Purpose:** What's this post about? (announcement, product, quote, etc.)
    🎨 **Style:** What vibe? (clean, bold, fun, professional)
    🌈 **Colors:** Preferred colors?
    💬 **Message:** Key text or message to include?

    Example: *"Bella's Cafe, new menu launch, warm appetizing style, orange/red, 'Try Our New Summer Menu!'"*""",

            "linkedin_cover": f"""Excellent! A LinkedIn cover will make your profile stand out. Please share:

    🏢 **Name/Brand:** What should be prominently featured?
    💼 **Title:** Your professional role or expertise?
    🎯 **Message:** What should visitors know about you?
    🎨 **Style:** Professional, creative, or industry-specific?
    🌈 **Colors:** Professional color preferences?

    Example: *"John Smith, Senior Developer, 'Building innovative web solutions,' modern professional, blue/gray"*"""
        }
        
        # Return specific template or generic one
        return fallback_templates.get(asset_type, f"""Awesome! I'll create a {asset_display} for you. To make it perfect, could you share:

    🏢 **Brand Name:** What's your brand called?
    🎯 **Purpose:** What's this {asset_display} for?
    🎨 **Style:** What vibe do you want?
    🌈 **Colors:** Any color preferences?
    👥 **Audience:** Who will see this?

    Feel free to share as much detail as you'd like!""")


    def provide_brand_consultancy(self, user_input: str) -> str:
        """Provide brand consultancy advice without generating assets"""
        
        consultancy_prompt = f"""
        The user is asking for brand consultancy advice. Provide expert recommendations.
        
        User request: "{user_input}"
        
        Analyze the request and provide detailed, professional advice with specific recommendations.
        
        Format your response as expert consultancy advice with:
        1. Direct answer to their question
        2. Specific recommendations with examples
        
        Be conversational but expert-level professional.
        """
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert brand consultant providing professional advice. Be specific, helpful, and authoritative."},
                    {"role": "user", "content": consultancy_prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            consultancy_response = response.choices[0].message.content.strip()
            print(f"[DEBUG] Provided consultancy advice for: {user_input}")
            
            return consultancy_response
            
        except Exception as e:
            print(f"[DEBUG] Consultancy error: {e}")
            return "I'd be happy to help with brand consultancy! Could you provide more details about what specific aspect of branding you'd like advice on?"
        
    def get_final_asset_type(self, query: str) -> str:
        """Get the final asset type through multiple detection methods"""
        
        # Method 1: Check if already detected from conversation
        conversation_type = getattr(self, 'detected_asset_type', None)
        if conversation_type and conversation_type != 'logo':
            print(f"[DEBUG] Using conversation detected type: {conversation_type}")
            return conversation_type
        
        # Method 2: Extract from current input
        self.extract_from_current_input(query)
        current_type = getattr(self, 'detected_asset_type', None)
        if current_type and current_type != 'logo':
            print(f"[DEBUG] Using current input detected type: {current_type}")
            return current_type
        
        # Method 3: Use GPT detection as fallback
        asset_info = self.detect_asset_type_and_specs(query)
        final_type = asset_info.get("type", "logo")
        print(f"[DEBUG] Using GPT detected type: {final_type}")
        
        return final_type

    def detect_asset_type_and_specs(self, user_input: str) -> dict:
        """Use GPT to intelligently detect asset type and specifications"""
        
        detection_prompt = f"""
        Analyze this user input and determine what they want.
        
        User input: "{user_input}"
        
        DETERMINE IF THIS IS:
        1. ASSET GENERATION REQUEST - User wants a specific visual asset created
        2. CONSULTANCY REQUEST - User wants advice/recommendations (no asset creation)
        
        CONSULTANCY INDICATORS:
        - "Pick colors for..."
        - "What colors work for..."  
        - "Recommend fonts for..."
        - "Best practices for..."
        - "Colors for [industry] company" (no specific brand name)
        
        ASSET GENERATION INDICATORS:
        - "Create logo for [Brand Name]"
        - "Generate Instagram post"  
        - "Design business card for [Company]"
        - Specific brand name mentioned
        
        Available asset types and dimensions:
        - logo: 1024x1024 (specific brand logos)
        - instagram_post: 1080x1080 (brand social posts)
        - linkedin_cover: 1584x396 (brand profile covers)
        - poster: 1080x1350 (brand marketing materials)
        - consultancy: N/A (advice/recommendations only)
        
        Return only JSON with detected type, don't give any text other than JSON:
        Format: """ + '{"type": "consultancy/logo/instagram_post/etc", "dimensions": "width_x_height or N/A", "confidence": "high/medium/low", "reasoning": "brief explanation"}'
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Distinguish between asset generation requests and consultancy requests. Always return valid JSON."},
                    {"role": "user", "content": detection_prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            detection_text = response.choices[0].message.content.strip()
            print(f"[DEBUG] Asset type detection result: {detection_text}")
            
            # Clean up any markdown formatting
            if detection_text.startswith('```json'):
                detection_text = detection_text.replace('```json', '').replace('```', '').strip()
            elif detection_text.startswith('```'):
                detection_text = detection_text.replace('```', '').strip()
            
            import json
            detection_result = json.loads(detection_text)
            
            # Validate the result
            asset_type = detection_result.get("type", "consultancy")
            dimensions = detection_result.get("dimensions", "N/A")
            confidence = detection_result.get("confidence", "medium")
            reasoning = detection_result.get("reasoning", "default detection")
            
            print(f"[DEBUG] Detected: {asset_type} ({dimensions}) - Confidence: {confidence}")
            print(f"[DEBUG] Reasoning: {reasoning}")
            
            return {
                "type": asset_type,
                "dimensions": dimensions,
                "confidence": confidence,
                "reasoning": reasoning
            }
            
        except Exception as e:
            print(f"[DEBUG] Asset type detection error: {e}")
            
            # Fallback: check for consultancy keywords
            consultancy_keywords = [
                "pick colors", "choose colors", "recommend colors", 
                "what colors", "best colors", "colors for",
                "pick fonts", "recommend fonts", "best fonts",
                "brand advice", "branding tips"
            ]
            
            user_lower = user_input.lower()
            if any(keyword in user_lower for keyword in consultancy_keywords):
                return {
                    "type": "consultancy",
                    "dimensions": "N/A",
                    "confidence": "high",
                    "reasoning": "consultancy keywords detected"
                }
            
            # Default fallback
            return {
                "type": "consultancy", 
                "dimensions": "N/A",
                "confidence": "low",
                "reasoning": "default consultancy fallback"
            }

    

    def generate_brand_asset_dalle(self, info: dict, asset_type: str, dimensions: str, user_context: str = ""):
        """Enhanced brand asset generation with dynamic messaging"""
        
        brand_name = info.get('brand_name', 'Brand')
        target_audience = info.get('target_audience', 'general audience')
        color_palette = info.get('color_palette', 'professional colors')
        brand_personality = info.get('brand_personality', 'professional')
        industry = info.get('industry', 'general business')
        logo_type = info.get('logo_type', 'modern')

        def generate_dynamic_asset_success_message(
    brand_name: str, 
    asset_type: str, 
    brand_personality: str,
    target_audience: str, 
    color_palette: str, 
    user_context: str = "",
    industry: str = "business"
) -> str:
            """Generate dynamic, conversational success messages using GPT for any asset type"""
            
            # Asset display names
            asset_displays = {
                "logo": "logo",
                "instagram_post": "Instagram post",
                "instagram_story": "Instagram story",
                "linkedin_cover": "LinkedIn cover", 
                "facebook_cover": "Facebook cover",
                "youtube_thumbnail": "YouTube thumbnail",
                "twitter_header": "Twitter header",
                "poster": "poster",
                "business_card": "business card",
                "brochure": "brochure",
                "letterhead": "letterhead",
                "web_banner": "website banner",
                "email_signature": "email signature",
                "flyer": "flyer",
                "thumbnail": "thumbnail"
            }
            
            asset_display = asset_displays.get(asset_type, asset_type.replace('_', ' '))
            
            # Asset-specific conversation starters and benefits
            asset_specific_info = {
                "logo": {
                    "benefits": ["versatile for all brand materials", "scalable for any size", "memorable brand identity"],
                    "next_steps": ["create matching business cards", "design social media profile pictures", "develop a complete brand package", "generate marketing materials"]
                },
                "instagram_post": {
                    "benefits": ["optimized for high engagement", "mobile-friendly design", "algorithm-favored format"],
                    "next_steps": ["create matching Instagram stories", "design a content series", "generate Facebook versions", "create LinkedIn adaptations"]
                },
                "linkedin_cover": {
                    "benefits": ["professional credibility boost", "networking advantage", "career-focused design"],
                    "next_steps": ["create matching business cards", "design email signatures", "generate presentation templates", "create professional social posts"]
                },
                "youtube_thumbnail": {
                    "benefits": ["maximized click-through potential", "search visibility optimized", "viewer attraction focused"],
                    "next_steps": ["create thumbnail series", "design channel art", "generate video promotional posts", "create Instagram story versions"]
                },
                "business_card": {
                    "benefits": ["professional networking tool", "brand consistency", "print-ready quality"],
                    "next_steps": ["design matching letterheads", "create email signatures", "generate business stationery", "develop brand templates"]
                },
                "poster": {
                    "benefits": ["high visual impact", "versatile marketing tool", "attention-grabbing design"],
                    "next_steps": ["create social media versions", "design smaller flyer formats", "generate digital adaptations", "create event promotions"]
                }
            }
            
            # Default fallback info
            default_info = {
                "benefits": ["professional brand representation", "high-quality design", "purpose-optimized format"],
                "next_steps": ["create variations", "design matching materials", "generate complementary assets", "develop brand consistency"]
            }
            
            asset_info = asset_specific_info.get(asset_type, default_info)
            
            # Context integration
            context_mention = ""
            if user_context and user_context.strip():
                context_mention = f"perfectly incorporating your specific requirements about {user_context[:50]}{'...' if len(user_context) > 50 else ''}"
            
            # GPT-powered dynamic message generation
            message_prompt = f"""
            Generate an enthusiastic, conversational success message for a {asset_display} that was just created.
            
            ASSET DETAILS:
            - Brand: {brand_name}
            - Asset Type: {asset_display}
            - Brand Personality: {brand_personality}
            - Target Audience: {target_audience}
            - Colors: {color_palette}
            - Industry: {industry}
            - User Context: {user_context if user_context else 'None'}
            
            KEY BENEFITS TO MENTION (pick 2-3):
            {', '.join(asset_info['benefits'])}
            
            CONVERSATION STARTERS (pick 2-3):
            {', '.join(asset_info['next_steps'])}
            
            REQUIREMENTS:
            1. Start with genuine excitement about the completed {asset_display}
            2. Mention 2-3 specific benefits or design elements that work well
            3. Reference the context if provided: {context_mention if context_mention else 'N/A'}
            4. Ask 2-3 engaging follow-up questions or suggest next steps
            5. Use emojis strategically (not overuse)
            6. Be conversational like ChatGPT - natural, not robotic
            7. Keep it 3-4 sentences with natural flow
            8. Include specific asset type benefits
            
            TONE: Enthusiastic designer who just completed an amazing project and is excited to show it off and discuss next steps.
            
            Generate a unique, natural response that doesn't sound templated.
            """
            
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": f"Generate enthusiastic, natural success messages for {asset_display} completions. Be conversational like ChatGPT, not robotic. Show genuine excitement about the design work."},
                        {"role": "user", "content": message_prompt}
                    ],
                    temperature=0.8,  # Higher temperature for natural variation
                    max_tokens=400
                )
                
                dynamic_message = response.choices[0].message.content.strip()
                print(f"[DEBUG] Generated dynamic success message for {asset_type}: {dynamic_message[:100]}...")
                
                return dynamic_message
                
            except Exception as e:
                print(f"[DEBUG] Dynamic message generation error: {e}")
                
                # Smart fallback with asset-specific templates
                fallback_templates = {
                    "logo": [
                        f"🎉 **Your {brand_name} logo is ready!** I love how the {brand_personality} design captures your brand essence for {target_audience}. The {color_palette} creates exactly the professional impact you need. Want to see this on business cards or create social media versions? 🎨",
                        
                        f"✨ **{brand_name} logo complete!** This {brand_personality} approach really works for {target_audience} - it's versatile, memorable, and perfectly branded. Should we create matching materials or explore some color variations? What's your next priority? 🚀"
                    ],
                    
                    "instagram_post": [
                        f"🎉 **Your {brand_name} Instagram post is ready!** This {brand_personality} design{' ' + context_mention if context_mention else ''} will definitely stand out in feeds. The {color_palette} really pops on mobile! Want to create matching stories or design a whole content series? 📱✨",
                        
                        f"✨ **Instagram post complete for {brand_name}!** Love how this captures your {brand_personality} vibe for {target_audience}. It's perfectly optimized for engagement. Should we create more posts in this style or adapt it for other platforms? 🚀"
                    ],
                    
                    "linkedin_cover": [
                        f"🎉 **Your {brand_name} LinkedIn cover is complete!** This {brand_personality} design will make your profile incredibly professional and memorable for {target_audience}. The {color_palette} builds perfect trust and credibility. Want matching business cards or email signatures? 💼✨",
                        
                        f"✨ **LinkedIn cover ready for {brand_name}!** The {brand_personality} approach perfectly represents your professional brand. This will definitely make you stand out to {target_audience}. Should we create presentation templates or other professional materials? 🚀"
                    ]
                }
                
                if asset_type in fallback_templates:
                    import random
                    return random.choice(fallback_templates[asset_type])
                
                # Generic fallback
                return f"🎨 **Your {brand_name} {asset_display} is ready!** I've created a {brand_personality} design{' ' + context_mention if context_mention else ''} that perfectly captures your brand for {target_audience}. The {color_palette} creates exactly the right impact! Want to create variations or matching brand materials? ✨"

        
        # ✅ ENHANCED ASSET PROMPTS - More sophisticated and context-aware
        def build_enhanced_prompt(asset_type: str, user_context: str) -> str:
            """Build sophisticated prompts for different asset types"""
            
            # Context extraction for better prompts
            context_elements = ""
            if user_context and user_context.strip():
                try:
                    context_prompt = f"""
                    Extract specific visual elements from this context for design: "{user_context}"
                    Return 2-3 specific elements that should be featured prominently.
                    Format: "element1, element2, element3"
                    """
                    
                    response = openai_client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "Extract specific visual design elements from user context."},
                            {"role": "user", "content": context_prompt}
                        ],
                        temperature=0.3,
                        max_tokens=100
                    )
                    
                    extracted = response.choices[0].message.content.strip()
                    context_elements = f" featuring {extracted}"
                    
                except Exception as e:
                    context_elements = f" about {user_context}"
            
            # Enhanced prompts for each asset type
            prompts = {
                "logo": f"Create a professional {logo_type} logo design for '{brand_name}'{context_elements}. Clean, scalable, memorable business branding. Vector-style design optimized for multiple sizes.",
                
                "instagram_post": f"Design an engaging Instagram post for '{brand_name}'{context_elements}. Square format (1080x1080), mobile-optimized, high engagement potential with clear visual hierarchy and thumb-stopping appeal.",
                
                "instagram_story": f"Create a vertical Instagram story for '{brand_name}'{context_elements}. Mobile-first design (1080x1920), story-optimized with engaging visual elements and clear messaging.",
                
                "linkedin_cover": f"Design a professional LinkedIn cover banner for '{brand_name}'{context_elements}. Corporate header format (1584x396), business networking optimized, professional credibility focus.",
                
                "facebook_cover": f"Create a Facebook cover photo for '{brand_name}'{context_elements}. Social banner format (1200x630), community-focused design with brand personality.",
                
                "youtube_thumbnail": f"Design a high-CTR YouTube thumbnail for '{brand_name}'{context_elements}. Bold, attention-grabbing format (1280x720) optimized for search visibility and click-through rates.",
                
                "twitter_header": f"Create a Twitter header banner for '{brand_name}'{context_elements}. Clean social header (1500x500), brand-focused design optimized for Twitter platform.",
                
                "poster": f"Design a promotional poster for '{brand_name}'{context_elements}. Marketing poster format with clear hierarchy, eye-catching design for promotional use.",
                
                "business_card": f"Create a professional business card for '{brand_name}'{context_elements}. Standard card format (1050x600), print-ready design with clear contact hierarchy.",
                
                "brochure": f"Design a business brochure for '{brand_name}'{context_elements}. Information layout, professional design for business materials and marketing.",
                
                "letterhead": f"Create company letterhead for '{brand_name}'{context_elements}. Business stationery format, corporate branding design for official communications.",
                
                "web_banner": f"Design a website banner for '{brand_name}'{context_elements}. Modern web format, hero image design optimized for digital platforms.",
                
                "email_signature": f"Create an email signature banner for '{brand_name}'{context_elements}. Professional footer design, clean and minimal for email communications.",
                
                "flyer": f"Design a marketing flyer for '{brand_name}'{context_elements}. Event/promotion format, attention-grabbing design for marketing distribution.",
                
                "thumbnail": f"Create a content thumbnail for '{brand_name}'{context_elements}. High-engagement format, optimized for content platforms and social sharing."
            }
            
            base_prompt = prompts.get(asset_type, prompts["logo"])
            
            # Add brand context
            complete_prompt = f"""{base_prompt}
            
    Brand Context:
    - Target Audience: {target_audience}
    - Brand Personality: {brand_personality} 
    - Industry: {industry}
    - Color Palette: {color_palette}
    - Style Direction: {logo_type}

    Design Requirements:
    - High quality, professional execution
    - Platform-optimized for {asset_type.replace('_', ' ')}
    - Brand-consistent visual identity
    - No text overlays unless specifically requested
    - Optimized for intended use case"""

            return complete_prompt
        
        # Build the enhanced prompt
        prompt = build_enhanced_prompt(asset_type, user_context)
        
        # Parse dimensions for DALL-E (existing logic)
        dalle_size = "1024x1024"
        if dimensions in ["1024x1024", "1792x1024", "1024x1792"]:
            dalle_size = dimensions
        elif "x" in dimensions:
            try:
                width, height = map(int, dimensions.split("x"))
                if width > height:
                    dalle_size = "1792x1024"
                elif height > width:
                    dalle_size = "1024x1792"
                else:
                    dalle_size = "1024x1024"
            except:
                dalle_size = "1024x1024"
        
        try:
            print(f"[DEBUG] Generating {asset_type} with enhanced prompts...")
            print(f"[DEBUG] Brand: {brand_name}")
            print(f"[DEBUG] Asset Type: {asset_type}")
            print(f"[DEBUG] Dimensions: {dimensions} -> DALL-E: {dalle_size}")
            print(f"[DEBUG] Enhanced Prompt: {prompt[:200]}...")
            
            # Generate with DALL-E 3
            result = openai_client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=dalle_size,
                quality="standard",
                n=1
            )
            
            image_url = result.data[0].url
            print(f"[DEBUG] Successfully generated {asset_type}!")
            
            # ✅ USE DYNAMIC SUCCESS MESSAGE FUNCTION
            dynamic_message = generate_dynamic_asset_success_message(
                brand_name=brand_name,
                asset_type=asset_type,
                brand_personality=brand_personality,
                target_audience=target_audience,
                color_palette=color_palette,
                user_context=user_context,
                industry=industry
            )
            
            return {
                "type": "asset_generated", 
                "message": dynamic_message,
                "image_url": image_url,
                "asset_type": asset_type,
                "dimensions": dimensions,
                "brand_info": info
            }
            
        except Exception as e:
            print(f"[DEBUG] Enhanced generation error: {str(e)}")
            return {
                "type": "error",
                "message": f"I encountered an issue generating your {asset_type.replace('_', ' ')}: {str(e)}. Let me try a different approach.",
                "image_url": None,
                "brand_info": info
            }
        
    

    def collect_asset_info(self, asset_type: str, missing_info: list, provided_info: dict) -> str:
        """Generate natural conversation responses using GPT instead of rigid templates"""
        
        # If we have enough info (brand_name + at least one other field), auto-complete and generate
        if provided_info.get("brand_name") and len(provided_info) >= 2:
            print("[DEBUG] Enough info provided, auto-completing and generating...")
            # Auto-complete missing fields
            auto_completed = self.intelligent_auto_complete(self.design_info.copy(), asset_type)
            for key, value in auto_completed.items():
                if not self.design_info.get(key):
                    self.design_info[key] = value
            self.save_brand_design()
            
            # Generate the asset
            dimensions = self.get_dimensions_for_asset_type(asset_type)
            asset_result = self.generate_brand_asset_dalle(self.design_info, asset_type, dimensions)
            
            if asset_result["type"] == "asset_generated":
                self.last_generated_image = asset_result["image_url"]
                return f"""ASSET_GENERATED|{asset_result['image_url']}|{asset_result['message']}"""
            else:
                return asset_result["message"]
        
        # Otherwise, ask for more info naturally using GPT
        prompt = f"""
        You are a friendly brand designer talking to a client who wants to create a {asset_type.replace('_', ' ')}.
        
        CURRENT SITUATION:
        - Asset they want: {asset_type.replace('_', ' ')}
        - Information they've provided: {provided_info}
        - Information still needed: {missing_info}
        
        Generate a natural, conversational response that:
        1. Asks for the most important missing information
        2. Sounds friendly and professional (like ChatGPT)
        3. Gives examples or options to help them respond
        4. Shows enthusiasm about their project
        
        Keep it conversational, not robotic. Don't use phrases like "I'd love to" repeatedly.
        Be creative and natural in your phrasing.
        """
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Generate natural, conversational responses like a real brand designer would. Be enthusiastic but not repetitive."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,  # Higher temperature for natural variation
                max_tokens=250
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"[DEBUG] GPT response error: {e}")
            # Simple fallback without rigid templates
            if missing_info:
                next_field = missing_info[0]
                return f"I'm excited to work on your {asset_type.replace('_', ' ')}! Could you tell me about your {next_field.replace('_', ' ')}?"
            else:
                return "Let me create that for you right now!"


    def get_dimensions_for_asset_type(self, asset_type: str) -> str:
        """Get appropriate dimensions for asset type"""
        
        dimensions_map = {
            "logo": "1024x1024",
            "poster": "1080x1350", 
            "instagram_post": "1080x1080",
            "instagram_story": "1080x1920",
            "linkedin_cover": "1584x396",
            "facebook_cover": "1200x630",
            "youtube_thumbnail": "1280x720",
            "business_card": "1050x600"
        }
        
        return dimensions_map.get(asset_type, "1024x1024")

    def intelligent_auto_complete(self, provided_info: dict, asset_type: str = "logo"):
        """Enhanced auto-completion with asset-type awareness"""
        
        known_info = ""
        missing_fields = []
        for key, value in provided_info.items():
            if value:
                known_info += f"{key}: {value}\n"
            else:
                missing_fields.append(key)
        
        if not known_info.strip():
            return provided_info
        
        completion_prompt = f"""
        Based on the following brand information, intelligently suggest appropriate values for the missing fields.
        The user wants to create a {asset_type.replace('_', ' ')}.

        Known information:
        {known_info}

        Please complete this JSON with intelligent defaults for the missing fields: {missing_fields}

        Guidelines based on asset type "{asset_type}":
        - For social media assets: Consider platform-specific audience behavior
        - For business materials: Prioritize professionalism and clarity
        - For marketing materials: Focus on eye-catching and memorable elements
        - For logos: Consider scalability and versatility
        
        Industry-specific suggestions:
        - Tech companies: Modern, minimalist, tech-friendly colors (blues, grays, greens)
        - Food/Restaurant: Warm, appetizing colors (reds, oranges, yellows)
        - Healthcare: Trust-building colors (blues, whites, greens)
        - Finance: Professional, trustworthy (navy, gray, gold)
        - Creative/Design: Vibrant, artistic (varied creative palette)

        Return ONLY a complete JSON object with all fields:
        {{
            "brand_name": "{provided_info.get('brand_name', 'Modern Brand')}",
            "logo_type": "suggest based on brand and asset type",
            "target_audience": "suggest based on brand and asset type",
            "color_palette": "suggest based on brand and asset type",
            "brand_personality": "suggest based on brand context",
            "industry": "suggest based on brand name",
            "preferred_fonts": "suggest based on brand personality"
        }}
        """
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert brand consultant with deep knowledge of asset-specific design requirements. Always return valid JSON."},
                    {"role": "user", "content": completion_prompt}
                ],
                temperature=0.3,
                max_tokens=400
            )
            
            completed_text = response.choices[0].message.content.strip()
            print(f"[DEBUG] Auto-completion result: {completed_text}")
            
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


    def handle_query(self, query: str):
        """Handle user query using LangChain agent with tools (prioritized)"""
        
        # Save user message to MongoDB
        if self.conversation_id and self.user_id:
            MongoDB.save_message(
                conversation_id=self.conversation_id,
                user_id=self.user_id,
                sender='user',
                text=query
            )
            print("Message saved")

        # Store user query in Pinecone
        store_in_pinecone(
        agent_type="brand-designer", 
        role="user", 
        text=query,
        user_id=self.user_id,
        conversation_id=self.conversation_id  # Add conversation_id
        )

        # Retrieve similar past queries
        past_results = retrieve_from_pinecone(query)
        if past_results.matches:
            print(f"[DEBUG] Similar past entries found: {past_results.matches}")

        try:
            # Use LangChain agent FIRST (which has access to tools)
            print(f"[DEBUG] Using LangChain agent with DALL-E tools")
            ai_response = self.agent.run(query)
            
            # CRITICAL FIX: Check if LangChain corrupted a LOGO_GENERATED response
            if self.last_generated_image and "ASSET_GENERATED|" not in ai_response:
                print(f"[DEBUG] LangChain corrupted the LOGO_GENERATED response")
                print(f"[DEBUG] Original response: {ai_response}")
                
                # Extract the image URL from the corrupted response
                if self.last_generated_image in ai_response or "[Logo](" in ai_response:
                    # Reconstruct the proper format
                    brand_name = self.design_info.get('brand_name', 'brand')
                    asset_type = getattr(self, 'detected_asset_type', 'design')
                    color_palette = self.design_info.get('color_palette', 'professional colors')
                    target_audience = self.design_info.get('target_audience', 'your audience')
        
                    # Reconstruct the proper format
                    success_message = f"I've created a professional design that perfectly captures your brand identity."
        
                    ai_response = f"ASSET_GENERATED|{self.last_generated_image}|{success_message}"
                    print(f"[DEBUG] Reconstructed proper format: {ai_response}")
            
        except Exception as e:
            print(f"LangChain agent error: {e}, falling back to OpenAI direct")
            
            # Build context for fallback
            context_info = ""
            if self.conversation_id and self.user_id:
                previous_messages = MongoDB.get_conversation_messages(self.conversation_id, self.user_id)
                if previous_messages:
                    previous_messages = previous_messages[:-1]  # Remove current message
                
                if previous_messages:
                    context_info = "Previous conversation context:\n"
                    for msg in previous_messages[-6:]:  # Last 6 messages
                        role = "User" if msg['sender'] == 'user' else "Assistant"
                        context_info += f"{role}: {msg['text']}\n"
                    context_info += "\nCurrent conversation:\n"
                print(f"Context: {context_info}")

            # Enhanced query with context
            if context_info:
                enhanced_query = f"{context_info}User: {query}\nAssistant:"
            else:
                enhanced_query = f"User: {query}\nAssistant:"

            # Fallback to OpenAI direct (without tools)
            try:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": enhanced_query}
                ]
                
                response = openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000
                )
                
                ai_response = response.choices[0].message.content
                print(f"[DEBUG] Using OpenAI direct approach (fallback)")
                
            except Exception as e2:
                print(f"Both LangChain and OpenAI failed: {e2}")
                ai_response = "I'm experiencing technical difficulties. Please try again in a moment."

        # Save agent response to MongoDB
        if self.conversation_id and self.user_id:
            MongoDB.save_message(
                conversation_id=self.conversation_id,
                user_id=self.user_id,
                sender='agent',
                text=ai_response,
                agent=self.agent_name
            )

        # Store in Pinecone
        store_in_pinecone(
        agent_type="brand-designer", 
        role="agent", 
        text=ai_response,
        user_id=self.user_id,
        conversation_id=self.conversation_id
        )
        
        return ai_response


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
            agent_type="brand-designer", 
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
            # Check if this is a design generation request
            wants_generation = self.detect_generation_intent(query)
            
            if wants_generation:
                # ✅ USE REAL THINKING-ENABLED ASSET GENERATION
                async for chunk in self.stream_asset_generation_with_real_thinking(query):
                    yield chunk
            else:
                # Stream regular conversation (keep existing method)
                async for chunk in self.stream_conversation_response(query):
                    yield chunk
                    
        except Exception as e:
            yield {
                "type": "error", 
                "message": f"Processing error: {str(e)}"
            }

    
    async def stream_asset_generation_with_real_thinking(self, query: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream asset generation with REAL model thinking - TRUE SEQUENTIAL EXECUTION"""
        try:
            if self.conversation_id and self.user_id:
                MongoDB.save_message(
                    conversation_id=self.conversation_id,
                    user_id=self.user_id,
                    sender='user',
                    text=query
                )
                print(f"[DEBUG] User message saved to MongoDB: {query}")
        
        # ✅ Store user query in Pinecone
            store_in_pinecone(
                agent_type="brand-designer", 
                role="user", 
                text=query,
                user_id=self.user_id,
                conversation_id=self.conversation_id
            )
            print(f"[DEBUG] User message stored in Pinecone")
        
            # ✅ STEP 1: REAL MODEL THINKING ABOUT THE REQUEST
            yield {
                "type": "thinking_start",
                "message": "🧠 Thinking...",
                "status": "thinking"
            }
            
            # ✅ WAIT FOR ACTUAL THINKING TO COMPLETE (not hardcoded time)
            thinking_result = await self.get_real_model_thinking(query)
            print("Model thinking completed:", thinking_result)
            
            yield {
                "type": "thinking_process",
                "message": "💭 Model's Real Thinking Process:",
                "thinking": thinking_result["thinking"],
                "reasoning": thinking_result["reasoning"],
                "analysis": thinking_result["analysis"],
                "plan": thinking_result["plan"],
                "status": "thinking_complete"
            }
            
            # ✅ SMALL UI DELAY (optional, for UX)
            await asyncio.sleep(0.3)  # Just for smooth UI transition

            # ✅ STEP 2: WEB SEARCH - STARTS ONLY AFTER THINKING IS DONE
            yield {
                "type": "tool_start",
                "tool_name": "Web Search Engine",
                "message": "🔍 Searching for design inspiration and references...",
                "status": "searching_web"
            }
            
            # ✅ WAIT FOR ACTUAL SEARCH TO COMPLETE
            search_keywords = self.convert_to_keywords(query)
            search_results = self.search_with_keywords(search_keywords)
            formatted_results = self.format_search_results(search_results)
            
            yield {
                "type": "web_search_complete",
                "tool_name": "Web Search Engine", 
                "message": f"✅ Found {len(formatted_results)} relevant articles and references",
                "data": {
                    "keywords": search_keywords,  # ✅ Frontend expects this
                    "results": formatted_results  # ✅ Frontend expects this
                },
                "status": "web_search_complete"
                }
            
            # ✅ SMALL UI DELAY
            await asyncio.sleep(0.3)
            
            # ✅ STEP 3: IMAGE SEARCH - STARTS ONLY AFTER WEB SEARCH IS DONE
            yield {
                "type": "tool_start",
                "tool_name": "Design Inspiration Finder",
                "message": "🎨 Searching Behance & Dribbble for design inspiration...",
                "status": "searching_inspiration"
            }
            
            # ✅ WAIT FOR ACTUAL IMAGE SEARCH TO COMPLETE
            inspiration_images = self.search_images(f"{search_keywords} design inspiration", num_results=8)
            
            yield {
                "type": "inspiration_images",
                "tool_name": "Design Inspiration Finder",
                "message": f"🎨 Found {len(inspiration_images)} design inspirations from Behance & Dribbble",
                "images": inspiration_images,
                "status": "inspiration_complete"
            }
            
            await asyncio.sleep(0.3)
            
            # ✅ STEP 4: BRAND EXTRACTION - STARTS ONLY AFTER INSPIRATION IS DONE
            yield {
                "type": "tool_start",
                "tool_name": "Brand Information Extractor",
                "message": "📋 Extracting brand information from your request...",
                "status": "extracting_info"
            }
            
            # ✅ WAIT FOR ACTUAL EXTRACTION THINKING TO COMPLETE
            extraction_thinking = await self.get_real_extraction_thinking(query)
            
            yield {
                "type": "thinking_process",
                "message": "🧠 Information Extraction Analysis:",
                "thinking": extraction_thinking["thinking"],
                "process": extraction_thinking["process"],
                "findings": extraction_thinking["findings"],
                "status": "extraction_thinking"
            }
            
            # ✅ PERFORM ACTUAL EXTRACTION (wait for completion)
            recent_messages = []
            if self.conversation_id and self.user_id:
                recent_messages = MongoDB.get_conversation_messages(self.conversation_id, self.user_id)
            
            if recent_messages:
                self.extract_brand_info_from_conversation(recent_messages)
            
            if query and query.strip():
                self.extract_from_current_input(query)
            
            extracted_info = {k: v for k, v in self.design_info.items() if v}
            
            yield {
                "type": "tool_result",
                "tool_name": "Brand Information Extractor",
                "message": f"✅ Extracted: {', '.join(extracted_info.keys()) if extracted_info else 'Basic information'}",
                "data": extracted_info,
                "status": "info_extracted"
            }
            
            # Check if we need more info before continuing
            if not self.design_info.get("brand_name"):
                asset_type = self.get_final_asset_type(query)

                comprehensive_questions = self.ask_comprehensive_asset_questions(asset_type)
                yield {
                    "type": "message",
                    "message": comprehensive_questions,
                    "status": "awaiting_input"
                }
                yield {
                "type": "complete",
                "status": "awaiting_input",
                "message": comprehensive_questions,
                "final_data": {
                    "search_results": formatted_results,
                    "search_keywords": search_keywords,
                    "inspiration_images": inspiration_images
                }
            }
                return
            
            await asyncio.sleep(0.3)
            
            # ✅ STEP 5: AUTO-COMPLETION - STARTS ONLY AFTER EXTRACTION IS DONE
            missing_info = [k for k, v in self.design_info.items() if not v]
            
            if missing_info:
                yield {
                    "type": "tool_start",
                    "tool_name": "Smart Auto-Completion",
                    "message": "🧠 Smart-completing missing brand details...",
                    "status": "auto_completing"
                }
                
                # ✅ WAIT FOR ACTUAL AUTO-COMPLETION TO COMPLETE
                auto_completed = self.intelligent_auto_complete(self.design_info.copy(), "logo")
                for key, value in auto_completed.items():
                    if not self.design_info.get(key):
                        self.design_info[key] = value
                
                completed_fields = [k for k in missing_info if self.design_info.get(k)]
                
                yield {
                    "type": "tool_result",
                    "tool_name": "Smart Auto-Completion", 
                    "message": f"✅ Completed: {', '.join(completed_fields) if completed_fields else 'Brand information'}",
                    "data": {k: self.design_info[k] for k in completed_fields if self.design_info.get(k)},
                    "status": "auto_completed"
                }
            
            await asyncio.sleep(0.3)
            
            # ✅ STEP 6: DESIGN THINKING - STARTS ONLY AFTER AUTO-COMPLETION IS DONE
            yield {
                "type": "tool_start",
                "tool_name": "Creative Design Process",
                "message": "🎨 Analyzing design strategy and creative approach...",
                "status": "design_thinking"
            }
            
            # ✅ WAIT FOR ACTUAL DESIGN THINKING TO COMPLETE
            design_thinking = await self.get_real_design_thinking(self.design_info, query)
            
            yield {
                "type": "thinking_process",
                "message": "🎨 Creative Design Strategy:",
                "thinking": design_thinking["thinking"],
                "creative_process": design_thinking["creative_process"],
                "design_decisions": design_thinking["decisions"],
                "prompt_strategy": design_thinking["prompt_strategy"],
                "status": "design_thinking_complete"
            }
            
            await asyncio.sleep(0.3)
            
            # ✅ STEP 7: ASSET GENERATION - STARTS ONLY AFTER DESIGN THINKING IS DONE
            yield {
                "type": "tool_start",
                "tool_name": "DALL-E 3 Asset Generator",
                "message": "✨ Generating your brand asset with DALL-E 3...",
                "status": "generating_asset"
            }
            
            # ✅ WAIT FOR ACTUAL ASSET GENERATION TO COMPLETE
            asset_result = self.generate_brand_asset_dalle(
                self.design_info, 
                "logo", 
                "1024x1024", 
                user_context=query
            )
            
            # At the end of your stream_asset_generation_with_real_thinking method:

            if asset_result["type"] == "asset_generated":
                yield {
                    "type": "tool_result",
                    "tool_name": "DALL-E 3 Asset Generator",
                    "message": "✅ Asset generated successfully!",
                    "status": "asset_generated"
                }

                await asyncio.sleep(0.3)
                
                # Final response with asset
                self.last_generated_image = asset_result["image_url"]
                
                yield {
                    "type": "asset_generated",
                    "message": asset_result["message"],
                    "image_url": asset_result["image_url"],
                    "brand_info": asset_result["brand_info"],
                    "status": "complete"
                }
                
                # ✅ ADD: Send completion signal
                yield {
                "type": "complete",
                "status": "complete",
                "final_data": {
                    "search_results": formatted_results,  # ✅ Include search results in completion
                    "search_keywords": search_keywords,
                    "inspiration_images": inspiration_images
                    }
                }
                
                # Save to MongoDB
                if self.conversation_id and self.user_id:
                    final_response = f"""ASSET_GENERATED|{asset_result['image_url']}|{asset_result['message']}"""
                    MongoDB.save_message(
                        conversation_id=self.conversation_id,
                        user_id=self.user_id,
                        sender='agent',
                        text=final_response,
                        agent=self.agent_name
                    )

                    store_in_pinecone(
                    agent_type="brand-designer", 
                    role="agent", 
                    text=final_response,
                    user_id=self.user_id,
                    conversation_id=self.conversation_id
                    )
                    search_results_data = {
                        "keywords": search_keywords,
                        "results": formatted_results
                    } if formatted_results else None
        
                    # Save message with search data
                    MongoDB.save_message_with_search_data(
                        conversation_id=self.conversation_id,
                        user_id=self.user_id,
                        sender='agent',
                        text=asset_result['message'],
                        agent=self.agent_name,
                        search_results=search_results_data,
                        inspiration_images=inspiration_images
                    )
                    store_in_pinecone(
                    agent_type="brand-designer", 
                    role="agent", 
                    text=asset_result['message'],
                    user_id=self.user_id,
                    conversation_id=self.conversation_id
                    )
            else:
                yield {
                    "type": "error",
                    "message": asset_result["message"],
                    "status": "generation_failed"
                }
                
                # ✅ ADD: Send completion signal even for errors
                yield {
                    "type": "complete",
                    "status": "error"
                }
                
        except Exception as e:
            yield {
                "type": "error",
                "message": f"Asset generation failed: {str(e)}",
                "status": "error"
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
        
        # ✅ Store user query in Pinecone
            store_in_pinecone(
                agent_type="brand-designer", 
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


    async def get_real_model_thinking(self, query: str) -> dict:
        """Get REAL model thinking using Groq reasoning model"""
        
        thinking_prompt = f"""
        <thinking>
        The user is asking me: "{query}"
        
        Let me think step by step about this brand design request:
        
        1. What exactly is the user asking for?
        2. What type of design asset do they want?
        3. What information will I need to create this effectively?
        4. What's my strategy for helping them achieve their goal?
        5. How can I make this process smooth and efficient?
        
        I need to analyze this carefully to provide the best possible service.
        </thinking>
        
        Analyze this user request for brand design work. Show your complete reasoning process.
        
        USER REQUEST: "{query}"
        
        Think through this step-by-step and show your reasoning.
        """
        
        try:
            response = self.groq_client.chat.completions.create(
                model=self.reasoning_model,  # DeepSeek R1 Distill Llama
                messages=[
                    {"role": "system", "content": "You are a professional brand designer. Think through client requests step-by-step, showing your complete reasoning process in <thinking> tags."},
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
                "analysis": "Analyzing the user's brand design request",
                "plan": "Developing strategy to help achieve their goals"
            }
            
        except Exception as e:
            print(f"Real thinking generation error: {e}")
            return {
                "thinking": f"I'm analyzing your request: {query}. Let me think about what type of design asset you need and what information I'll require to create something amazing for your brand.",
                "reasoning": "Processing user request for brand design assistance",
                "analysis": "User wants brand design help",
                "plan": "Gather brand information and create the requested asset"
            }

    async def get_real_extraction_thinking(self, query: str) -> dict:
        """Show REAL thinking process for information extraction using reasoning model"""
        
        extraction_prompt = f"""
        <thinking>
        I need to extract brand information from this request: "{query}"
        
        Let me think carefully:
        1. What brand information can I identify directly from this text?
        2. What clues about brand personality, industry, or style preferences are present?
        3. What's missing that I'll need to ask for?
        4. What can I intelligently infer from context?
        5. How should I prioritize the information I gather?
        
        I need to be thorough but efficient in my extraction process.
        </thinking>
        
        Extract brand information from this request, showing your reasoning process.
        
        REQUEST: "{query}"
        
        Think through your extraction strategy step-by-step.
        """
        
        try:
            response = self.groq_client.chat.completions.create(
                model=self.reasoning_model,
                messages=[
                    {"role": "system", "content": "Analyze text for brand information extraction. Show your complete reasoning process in <thinking> tags."},
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
                "findings": "Analyzing brand elements in the request"
            }
                
        except Exception as e:
            return {
                "thinking": f"I'm examining this request: {query}. I need to look for brand name, style preferences, target audience, and any specific requirements mentioned.",
                "process": f"Extracting information from: {query}",
                "findings": "Looking for brand name, style preferences, and requirements"
            }

    async def get_real_design_thinking(self, design_info: dict, user_context: str) -> dict:
        """Show REAL creative thinking process for design generation using reasoning model"""
        
        design_prompt = f"""
        <thinking>
        I'm about to create a design with this information:
        Brand Info: {design_info}
        User Context: "{user_context}"
        
        Let me think through the creative process:
        1. What visual style best represents this brand based on the information?
        2. How do I translate brand personality into visual elements?
        3. What prompt strategy will create the best DALL-E result?
        4. What design principles should I prioritize?
        5. How can I ensure this meets the user's expectations?
        
        This is the creative decision-making phase.
        </thinking>
        
        Plan the creative design process for this brand asset.
        
        BRAND INFO: {design_info}
        USER CONTEXT: "{user_context}"
        
        Show your creative reasoning step-by-step.
        """
        
        try:
            response = self.groq_client.chat.completions.create(
                model=self.reasoning_model,
                messages=[
                    {"role": "system", "content": "Think through the creative design process like a professional brand designer. Show your reasoning in <thinking> tags."},
                    {"role": "user", "content": design_prompt}
                ],
                temperature=0.4,
                max_tokens=800
            )
            
            thinking_text = response.choices[0].message.content.strip()
            
            # Extract thinking and response
            thinking_match = re.search(r'<thinking>(.*?)</thinking>', thinking_text, re.DOTALL)
            thinking_content = thinking_match.group(1).strip() if thinking_match else thinking_text
            
            response_match = re.search(r'</thinking>\s*(.*)', thinking_text, re.DOTALL)
            creative_content = response_match.group(1).strip() if response_match else thinking_text
            
            return {
                "thinking": thinking_content,
                "creative_process": creative_content,
                "decisions": "Focusing on brand consistency and visual impact",
                "prompt_strategy": "Creating detailed prompt for optimal DALL-E results"
            }
                
        except Exception as e:
            return {
                "thinking": "I'm considering how to translate this brand identity into visual design. I need to balance creativity with brand requirements and ensure the result serves its intended purpose.",
                "creative_process": "Translating brand identity into visual design",
                "decisions": "Balancing creativity with brand requirements",
                "prompt_strategy": "Optimizing prompt for brand-appropriate results"
            }


def get_brand_designer_agent(user_id: str = None, conversation_id: str = None):
    return BrandDesignerAgent(user_id, conversation_id)