import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool, AgentType
from core.config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENV
from core.database import MongoDB
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime
# ---------------------------
# Pinecone Setup (same as before)
# ---------------------------
pinecone = Pinecone(api_key=PINECONE_API_KEY)
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

def store_in_pinecone(agent_type: str, role: str, text: str):
    """Store only embedding in Pinecone (no raw text)."""
    vector = embed_text(text)
    vector_id = f"{agent_type}-{role}-{hash(text)}"
    pinecone_index.upsert([(vector_id, vector)])

def retrieve_from_pinecone(query: str, top_k: int = 3):
    """Retrieve most relevant embeddings for a query."""
    query_vector = embed_text(query)
    results = pinecone_index.query(vector=query_vector, top_k=top_k, include_values=False)
    return results

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
        
        # Build conversation text - include ALL messages, not just recent ones for brand name extraction
        conversation_text = ""
        for msg in messages:  # Use ALL messages to catch brand names mentioned earlier
            role = "User" if msg['sender'] == 'user' else "Assistant"
            conversation_text += f"{role}: {msg['text']}\n"
        
        if not conversation_text.strip():
            print("[DEBUG] No conversation history to extract from")
            return
        
        print(f"[DEBUG] Extracting from conversation: {conversation_text}")
        
        # Enhanced extraction prompt that looks for brand names throughout the conversation
        extraction_prompt = f"""
        Analyze this FULL conversation and extract brand information. Look for brand names mentioned ANYWHERE in the conversation.
        
        Conversation:
        {conversation_text}

        IMPORTANT RULES:
        1. Look for brand names mentioned ANYWHERE in the conversation (not just the last message)
        2. If user mentions "other brand", "another brand", "different brand" - they want to work on a NEW brand
        3. If user gives a specific brand name, use that exact name
        4. If working on a new brand, RESET all fields except what's explicitly mentioned
        5. Look for asset type clues (poster, logo, banner, instagram, linkedin, etc.)

        ASSET TYPE MAPPING:
        - "instagram poster" OR "instagram post" → "instagram_post"  
        - "linkedin poster" OR "linkedin cover" → "linkedin_cover"
        - "facebook poster" OR "facebook cover" → "facebook_cover"
        - "logo design" OR "logo" → "logo"
        - Generic "poster" without platform → "poster"

        Extract these fields by looking through the ENTIRE conversation:
        - brand_name: The brand name mentioned ANYWHERE (look carefully!)
        - logo_type: Style mentioned for ANY asset type
        - target_audience: Who they're targeting
        - color_palette: Colors mentioned
        - asset_type: What type of asset they want (logo, instagram_post, linkedin_cover, poster, etc.)
        - is_new_brand: true if they mentioned "other brand", "another brand", etc.

        CRITICAL: Return ONLY valid JSON, no markdown formatting, no explanations.
        Format: {{"brand_name": "exact name found in conversation or null", "logo_type": "extracted style or null", "target_audience": "extracted audience or null", "color_palette": "extracted colors or null", "asset_type": "logo/instagram_post/linkedin_cover/poster/etc or null", "is_new_brand": true/false}}
        """
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Extract brand information from the FULL conversation. Look for brand names mentioned anywhere. Return ONLY valid JSON with no markdown formatting."},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            extracted_text = response.choices[0].message.content.strip()
            print(f"[DEBUG] GPT extraction result: {extracted_text}")
            
            # Clean up any markdown formatting
            if extracted_text.startswith('```json'):
                extracted_text = extracted_text.replace('```json', '').replace('```', '').strip()
            elif extracted_text.startswith('```'):
                extracted_text = extracted_text.replace('```', '').strip()
            
            import json
            extracted_info = json.loads(extracted_text)
            
            # If it's a new brand, reset design_info
            if extracted_info.get("is_new_brand"):
                print("[DEBUG] New brand detected, resetting design_info")
                self.design_info = {
                    "brand_name": None,
                    "logo_type": None,
                    "target_audience": None,
                    "color_palette": None,
                    "brand_personality": None,
                    "industry": None,
                    "preferred_fonts": None,
                }
            
            # Update design_info with extracted information
            for key, value in extracted_info.items():
                if key in self.design_info and value and value.lower() not in ["null", "", "none"]:
                    self.design_info[key] = value
                    print(f"[DEBUG] Set {key}: {value}")
            
            # Store asset type separately
            if extracted_info.get("asset_type"):
                self.detected_asset_type = extracted_info["asset_type"]
                print(f"[DEBUG] Detected asset type from conversation: {self.detected_asset_type}")
            
            print(f"[DEBUG] Updated design_info: {self.design_info}")
            
        except Exception as e:
            print(f"[DEBUG] Extraction error: {e}")
            if 'extracted_text' in locals():
                print(f"[DEBUG] Raw extraction text: {extracted_text}")

            # Enhanced fallback - manually search for brand names in conversation
            print("[DEBUG] Using fallback brand name extraction")
            for msg in messages:
                text = msg['text'].lower()
                
                # Look for patterns like "brand name is X" or "my brand is X"
                import re
                brand_patterns = [
                    r"brand name is (\w+)",
                    r"my brand is (\w+)", 
                    r"brand called (\w+)",
                    r"company name is (\w+)",
                    r"business name is (\w+)",
                    r"for (\w+)\b",  # Simple pattern for brand names
                ]
                
                for pattern in brand_patterns:
                    match = re.search(pattern, text)
                    if match:
                        brand_name = match.group(1)
                        if brand_name not in ['a', 'the', 'my', 'our', 'is', 'are', 'logo', 'design']:
                            self.design_info['brand_name'] = brand_name
                            print(f"[DEBUG] Fallback extracted brand name: {brand_name}")
                            break
                
                # Look for asset type mentions
                if 'logo' in text:
                    self.detected_asset_type = 'logo'
                    print(f"[DEBUG] Fallback detected asset type: logo")
    
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


    def extract_from_current_input(self, user_input: str):
        """Extract brand info from current input with better brand switching detection"""
        
        current_info_text = ""
        if any(v for v in self.design_info.values()):
            current_info_text = f"Currently working on:\n"
            for key, value in self.design_info.items():
                if value:
                    current_info_text += f"- {key}: {value}\n"
            current_info_text += "\n"
        
        # FIXED: No f-string with JSON braces - use regular string concatenation
        extraction_prompt = f"""
        {current_info_text}User just said: "{user_input}"
        
        CRITICAL: If user mentions "other brand", "another brand", "different brand", they want to start fresh with a NEW brand.
        
        Analyze this input and extract:
        1. Asset type (poster, logo, linkedin_cover, instagram_post, etc.)
        2. Brand information (name, style, colors, audience)
        3. Whether this is for a new/different brand
        
        IMPORTANT ASSET TYPE MAPPING:
        - "instagram poster" OR "instagram post" → "instagram_post"
        - "linkedin poster" OR "linkedin cover" OR "linkedin banner" → "linkedin_cover"
        - "facebook poster" OR "facebook cover" → "facebook_cover"
        - Generic "poster" without platform → "poster"
        
        Return ONLY valid JSON with no markdown.
        Examples:
        - "generate instagram poster" → """ + '{"asset_type": "instagram_post", "brand_name": null, "is_new_brand": false}' + """
        - "create instagram post" → """ + '{"asset_type": "instagram_post", "brand_name": null, "is_new_brand": false}' + """
        - "generate poster for linkedin" → """ + '{"asset_type": "linkedin_cover", "brand_name": null, "is_new_brand": false}' + """
        - "create linkedin cover" → """ + '{"asset_type": "linkedin_cover", "brand_name": null, "is_new_brand": false}' + """
        - "generate poster for my other brand" → """ + '{"asset_type": "poster", "brand_name": null, "is_new_brand": true}' + """
        
        Return format: """ + '{"brand_name": "exact name or null", "logo_type": "extracted style or null", "target_audience": "extracted audience or null", "color_palette": "extracted colors or null", "asset_type": "instagram_post/linkedin_cover/poster/logo/etc or null", "is_new_brand": true/false, "brand_personality": "extracted personality or null", "industry": "extracted industry or null"}'
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Extract asset type and brand info. Map 'instagram poster' to 'instagram_post' and 'linkedin poster' to 'linkedin_cover'. Return ONLY valid JSON with no markdown."},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            extracted_text = response.choices[0].message.content.strip()
            print(f"[DEBUG] Current input extraction: {extracted_text}")
            
            # Clean up markdown formatting
            if extracted_text.startswith('```json'):
                extracted_text = extracted_text.replace('```json', '').replace('```', '').strip()
            elif extracted_text.startswith('```'):
                extracted_text = extracted_text.replace('```', '').strip()
            
            import json
            extracted_info = json.loads(extracted_text)
            
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
        """Smart brand asset generation supporting multiple asset types"""
        
        print(f"[DEBUG] smart_asset_generator called with input: '{user_input}'")
        print(f"[DEBUG] Initial design_info: {self.design_info}")
        
        # Store the original user input as context
        self.user_context = user_input
        
        # Initialize detected asset type
        self.detected_asset_type = None
        conversation_detected_type = None
        
        # Clean up user input for processing but keep original for context
        processed_input = user_input
        if user_input.startswith('{"') and user_input.endswith('}'):
            try:
                import json
                parsed_input = json.loads(user_input)
                processed_input = parsed_input.get("description", "generate asset")
            except:
                pass
        
        # Get conversation context  
        recent_messages = []
        if self.conversation_id and self.user_id:
            messages = MongoDB.get_conversation_messages(self.conversation_id, self.user_id)
            recent_messages = messages
            print(f"[DEBUG] Found {len(recent_messages)} recent messages")
        
        # Extract information (this now handles brand switching)
        if recent_messages:
            self.extract_brand_info_from_conversation(recent_messages)
            conversation_detected_type = getattr(self, 'detected_asset_type', None)
            print(f"[DEBUG] Conversation detected type: {conversation_detected_type}")
        
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
        
        # Check if brand name is needed
        if not self.design_info.get("brand_name"):
            return f"I'd love to create a {asset_type.replace('_', ' ')} for you! What's the name of your brand or business?"
        
        # Save updated design_info
        self.save_brand_design()
        
        print(f"[DEBUG] Current design_info after extraction: {self.design_info}")
        
        # Check missing information
        missing_info = [k for k, v in self.design_info.items() if not v]
        provided_info = {k: v for k, v in self.design_info.items() if v}
        
        print(f"[DEBUG] Missing info: {missing_info}")
        print(f"[DEBUG] Provided info: {provided_info}")
        
        # Detect generation intent
        wants_generation = self.detect_generation_intent(processed_input) if processed_input else False
        
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
            # Pass the original user input as context
            asset_result = self.generate_brand_asset_dalle(self.design_info, asset_type, dimensions, user_context=user_input)
            
            if asset_result["type"] == "asset_generated":
                self.last_generated_image = asset_result["image_url"]
                return f"""ASSET_GENERATED|{asset_result['image_url']}|{asset_result['message']}"""
            else:
                return asset_result["message"]
        
        # Asset-specific information collection
        return self.collect_asset_info(asset_type, missing_info, provided_info)
    
    def detect_asset_type_and_specs(self, user_input: str) -> dict:
        """Use GPT to intelligently detect asset type and specifications"""
        
        # FIXED: No f-string with JSON braces
        detection_prompt = f"""
        Analyze this user input and determine what type of visual asset they want to create.
        
        User input: "{user_input}"
        
        Available asset types and their optimal dimensions:
        
        LOGOS:
        - logo: 1024x1024 (general logos, brand marks)
        
        SOCIAL MEDIA:
        - instagram_post: 1080x1080 (square posts)
        - instagram_story: 1080x1920 (vertical stories)
        - linkedin_cover: 1584x396 (profile banner)
        - facebook_cover: 1200x630 (page cover)
        - youtube_thumbnail: 1280x720 (video thumbnails)
        - twitter_header: 1500x500 (profile header)
        
        MARKETING:
        - poster: 1080x1350 (promotional posters, flyers)
        - brochure: 1080x1350 (informational materials)
        
        BUSINESS:
        - business_card: 1050x600 (contact cards)
        - letterhead: 1080x1400 (company stationery)
        
        WEB:
        - web_banner: 1200x600 (website headers)
        - email_signature: 600x200 (email footers)
        
        Rules for detection:
        1. Look for specific mentions (e.g., "Instagram post" → instagram_post)
        2. Consider context clues (e.g., "social media graphic" could be instagram_post)
        3. If unclear, default to "logo" as the most common request
        4. Consider platform-specific keywords (LinkedIn, Facebook, etc.)
        5. Marketing terms like "flyer", "promotional" → poster
        
        Return JSON with the detected asset type and dimensions.
        
        Examples:
        - "create Instagram story" → """ + '{"type": "instagram_story", "dimensions": "1080x1920", "confidence": "high", "reasoning": "explicitly mentioned Instagram story"}' + """
        - "design a poster" → """ + '{"type": "poster", "dimensions": "1080x1350", "confidence": "high", "reasoning": "explicitly mentioned poster"}' + """
        - "make a social graphic" → """ + '{"type": "instagram_post", "dimensions": "1080x1080", "confidence": "medium", "reasoning": "social graphic typically means Instagram post"}' + """
        
        Format: """ + '{"type": "detected_asset_type", "dimensions": "width_x_height", "confidence": "high/medium/low", "reasoning": "brief explanation of why this type was chosen"}'
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at detecting visual asset types from user requests. Always return valid JSON with the exact format specified."},
                    {"role": "user", "content": detection_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent detection
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
            asset_type = detection_result.get("type", "logo")
            dimensions = detection_result.get("dimensions", "1024x1024")
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
            
            # Fallback to simple keyword detection
            user_lower = user_input.lower()
            
            # Define asset mappings as a simple fallback
            asset_keywords = {
                "poster": ["poster", "flyer", "promotional"],
                "instagram_post": ["instagram post", "ig post", "insta post"],
                "instagram_story": ["instagram story", "ig story", "insta story"], 
                "linkedin_cover": ["linkedin cover", "linkedin banner"],
                "youtube_thumbnail": ["youtube thumbnail", "yt thumbnail"],
                "business_card": ["business card", "visiting card"],
                "logo": ["logo", "brand mark", "symbol"]  # Default
            }
            
            # Simple keyword matching as fallback
            for asset_type, keywords in asset_keywords.items():
                if any(keyword in user_lower for keyword in keywords):
                    dimensions_map = {
                        "logo": "1024x1024",
                        "poster": "1080x1350", 
                        "instagram_post": "1080x1080",
                        "instagram_story": "1080x1920",
                        "linkedin_cover": "1584x396",
                        "youtube_thumbnail": "1280x720",
                        "business_card": "1050x600"
                    }
                    
                    return {
                        "type": asset_type,
                        "dimensions": dimensions_map.get(asset_type, "1024x1024"),
                        "confidence": "low",
                        "reasoning": "fallback keyword detection"
                    }
            
            # Ultimate fallback
            return {
                "type": "logo", 
                "dimensions": "1024x1024",
                "confidence": "low",
                "reasoning": "default fallback"
            }

    def generate_brand_asset_dalle(self, info: dict, asset_type: str, dimensions: str, user_context: str = ""):
        """Generate various brand assets with DALL-E using dynamic prompts that incorporate user context"""
        
        brand_name = info.get('brand_name', 'Brand')
        target_audience = info.get('target_audience', 'general audience')
        color_palette = info.get('color_palette', 'professional colors')
        brand_personality = info.get('brand_personality', 'professional')
        industry = info.get('industry', 'general business')
        logo_type = info.get('logo_type', 'modern')
        
        # Build dynamic prompt based on user context and asset type
        def build_dynamic_prompt(base_style: str, user_context: str) -> str:
            """Build a dynamic prompt that incorporates user's specific context"""
            
            # Base prompt structure
            base_prompt = f"Create a {base_style} for '{brand_name}'"
            
            # Add specific context if provided
            context_section = ""
            if user_context and user_context.strip():
                # Extract key context elements
                context_prompt = f"""
                Analyze this user context and extract key visual elements that should be included in the design:
                "{user_context}"
                
                Return 2-3 specific visual elements or messages that should be prominently featured in the design.
                Format: "Element 1, Element 2, Element 3"
                
                Examples:
                - "500 followers milestone" → "500 followers celebration, milestone achievement, social media success"
                - "new product launch" → "new product showcase, launch announcement, excitement"
                - "hiring designers" → "hiring call, designer wanted, join our team"
                """
                
                try:
                    response = openai_client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "Extract specific visual elements from user context for design purposes. Be concise and specific."},
                            {"role": "user", "content": context_prompt}
                        ],
                        temperature=0.3,
                        max_tokens=100
                    )
                    
                    extracted_elements = response.choices[0].message.content.strip()
                    context_section = f" featuring {extracted_elements}"
                    print(f"[DEBUG] Extracted context elements: {extracted_elements}")
                    
                except Exception as e:
                    print(f"[DEBUG] Context extraction error: {e}")
                    # Fallback: use original context
                    context_section = f" about {user_context}"
            
            # Asset-specific styling
            style_guidelines = {
                "logo": f"Professional logo design. Style: {logo_type}. Clean, scalable, memorable design.",
                "instagram_post": f"Instagram post design with square format optimized for mobile. Eye-catching, social media friendly design with clear visual hierarchy.",
                "instagram_story": f"Instagram story graphic with vertical mobile format. Engaging, thumb-stopping design optimized for stories.",
                "linkedin_cover": f"LinkedIn profile cover banner with professional business format. Corporate-appropriate design.",
                "facebook_cover": f"Facebook cover photo with social media banner format. Engaging, social-friendly design.",
                "youtube_thumbnail": f"YouTube thumbnail with high click-through appeal. Bold, attention-grabbing design.",
                "twitter_header": f"Twitter header banner with clean, professional social media design.",
                "poster": f"Promotional poster with clear message hierarchy. Eye-catching, informative design.",
                "brochure": f"Business brochure with information layout. Professional, organized design.",
                "business_card": f"Professional business card with contact card layout. Clean, readable, professional design.",
                "letterhead": f"Company letterhead with business stationery branding. Professional, corporate design.",
                "web_banner": f"Website banner or hero image. Modern, web-optimized design.",
                "email_signature": f"Email signature banner with professional footer design. Clean, minimal, professional design."
            }
            
            style_guide = style_guidelines.get(asset_type, style_guidelines["logo"])
            
            # Build complete prompt
            full_prompt = f"{base_prompt}{context_section}. {style_guide} Target audience: {target_audience}. Colors: {color_palette}. Brand personality: {brand_personality}."
            
            # Add platform-specific optimizations
            platform_optimizations = {
                "instagram_post": " Include engaging visual elements that perform well on Instagram feeds.",
                "instagram_story": " Design should be mobile-first and story-friendly.",
                "linkedin_cover": " Professional and business-appropriate for LinkedIn platform.",
                "youtube_thumbnail": " Bold text and high contrast elements for thumbnail visibility.",
                "business_card": " Print-ready design with clear contact information hierarchy."
            }
            
            if asset_type in platform_optimizations:
                full_prompt += platform_optimizations[asset_type]
            
            return full_prompt
        
        # Generate dynamic prompt
        asset_names = {
            "logo": "professional logo design",
            "instagram_post": "Instagram post design",
            "instagram_story": "Instagram story graphic",
            "linkedin_cover": "LinkedIn profile cover banner",
            "facebook_cover": "Facebook cover photo",
            "youtube_thumbnail": "YouTube thumbnail design",
            "twitter_header": "Twitter header banner",
            "poster": "promotional poster design",
            "brochure": "business brochure design",
            "business_card": "professional business card",
            "letterhead": "company letterhead design",
            "web_banner": "website banner design",
            "email_signature": "email signature banner"
        }
        
        base_style = asset_names.get(asset_type, "professional design")
        prompt = build_dynamic_prompt(base_style, user_context)
        
        # Parse dimensions for DALL-E
        dalle_size = "1024x1024"  # Default
        if dimensions in ["1024x1024", "1792x1024", "1024x1792"]:
            dalle_size = dimensions
        elif "x" in dimensions:
            width, height = map(int, dimensions.split("x"))
            if width > height:
                dalle_size = "1792x1024"
            elif height > width:
                dalle_size = "1024x1792"
            else:
                dalle_size = "1024x1024"
        
        try:
            print(f"[DEBUG] Generating {asset_type} with DALL-E...")
            print(f"[DEBUG] Requested dimensions: {dimensions}, Using DALL-E size: {dalle_size}")
            print(f"[DEBUG] Dynamic Prompt: {prompt}")
            
            result = openai_client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=dalle_size,
                quality="standard",
                n=1
            )
            
            image_url = result.data[0].url
            print(f"[DEBUG] Successfully generated {asset_type}!")
            print(f"[DEBUG] Image URL: {image_url}")
            
            # Create success message
            asset_display_names = {
                "logo": "logo",
                "instagram_post": "Instagram post",
                "instagram_story": "Instagram story", 
                "linkedin_cover": "LinkedIn cover",
                "facebook_cover": "Facebook cover",
                "youtube_thumbnail": "YouTube thumbnail",
                "twitter_header": "Twitter header",
                "poster": "poster",
                "brochure": "brochure", 
                "business_card": "business card",
                "letterhead": "letterhead",
                "web_banner": "website banner",
                "email_signature": "email signature"
            }
            
            asset_display_name = asset_display_names.get(asset_type, "design")
            
            # Include user context in success message if provided
            context_message = ""
            if user_context and user_context.strip():
                context_message = f" incorporating your specific requirements"
            
            success_message = f"🎨 **Your {brand_name} {asset_display_name} is ready!**\n\nI've created a {brand_personality} design{context_message} that perfectly captures your brand identity for {target_audience}. The design uses {color_palette} to create a memorable look."
            
            # Add platform-specific tips
            platform_tips = {
                "instagram_post": "\n\n💡 **Tip**: Perfect for Instagram feed posts and can be used on other social platforms too!",
                "instagram_story": "\n\n💡 **Tip**: Optimized for Instagram Stories - great for engagement and brand awareness!",
                "linkedin_cover": "\n\n💡 **Tip**: This will make your LinkedIn profile stand out professionally!",
                "youtube_thumbnail": "\n\n💡 **Tip**: Designed to maximize click-through rates on YouTube!",
                "business_card": "\n\n💡 **Tip**: Print at 300 DPI for best quality results!",
            }
            
            if asset_type in platform_tips:
                success_message += platform_tips[asset_type]
            
            return {
                "type": "asset_generated",
                "message": success_message,
                "image_url": image_url,
                "asset_type": asset_type,
                "dimensions": dimensions,
                "brand_info": info
            }
            
        except Exception as e:
            print(f"[DEBUG] DALL-E generation error: {str(e)}")
            return {
                "type": "error",
                "message": f"I encountered an issue generating your {asset_type.replace('_', ' ')}: {str(e)}. Let me try again with different specifications.",
                "image_url": None,
                "brand_info": info
            }
    
    def collect_asset_info(self, asset_type: str, missing_info: list, provided_info: dict) -> str:
        """Collect information based on asset type"""
        
        asset_display = asset_type.replace('_', ' ')
        
        if "brand_name" in missing_info:
            return f"I'd love to create a {asset_display} for you! What's the name of your brand or business?"
        
        elif "target_audience" in missing_info:
            return f"Great! For **{provided_info['brand_name']}**, who is your target audience for this {asset_display}? (e.g., young professionals, families, tech enthusiasts)"
        
        elif "color_palette" in missing_info:
            return f"Perfect! Any color preferences for your {asset_display}? You can mention specific colors, a mood, or say 'surprise me'!"
        
        elif asset_type == "logo" and "logo_type" in missing_info:
            return f"What style of logo are you thinking for **{provided_info['brand_name']}**? For example:\n\n• **Text-based** - stylized company name\n• **Icon-based** - symbol or graphic\n• **Combination** - text + icon together\n\nOr describe your preferred style!"
        
        else:
            # Auto-complete and generate
            auto_completed = self.intelligent_auto_complete(self.design_info.copy(), asset_type)
            for key, value in auto_completed.items():
                if not self.design_info.get(key):
                    self.design_info[key] = value
            self.save_brand_design()
            
            asset_result = self.generate_brand_asset_dalle(self.design_info, asset_type, "1024x1024")
            if asset_result["type"] == "asset_generated":
                self.last_generated_image = asset_result["image_url"]
                return f"""ASSET_GENERATED|{asset_result['image_url']}|{asset_result['message']}"""
            else:
                return asset_result["message"]

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
        messages = MongoDB.get_conversation_messages(self.conversation_id, self.user_id)
        for msg in messages:
            if msg['sender'] == 'user':
                self.memory.chat_memory.add_user_message(msg['text'])
            else:
                self.memory.chat_memory.add_ai_message(msg['text'])

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

        # Store user query in Pinecone
        store_in_pinecone("brand-designer", "user", query)

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
        store_in_pinecone("brand-designer", "assistant", ai_response)
        
        return ai_response
    
    

    def handle_query_with_context(self, query: str, previous_messages: list = None, context: str = None):
        """Handle user query with conversation context"""
        
        # Save user message to MongoDB
        if self.conversation_id and self.user_id:
            MongoDB.save_message(
                conversation_id=self.conversation_id,
                user_id=self.user_id,
                sender='user',
                text=query
            )

        # Build conversation context for better responses
        context_info = ""
        if context:
            context_info = f"Context: {context}\n\n"
        
        if previous_messages:
            context_info += "Previous conversation:\n"
            for msg in previous_messages[-5:]:  # Use last 5 messages for context
                role = msg.get("role", "user")
                content = msg.get("content", "")
                context_info += f"{role.title()}: {content}\n"
            context_info += "\n"

        # Enhanced query with context
        enhanced_query = f"{context_info}Current message: {query}"

        # Store user query in Pinecone
        store_in_pinecone("brand-designer", "user", query)

        # Retrieve similar past queries for additional context
        past_results = retrieve_from_pinecone(query)
        if past_results.matches:
            print(f"[DEBUG] Similar past entries found: {past_results.matches}")

        # Use OpenAI directly for better context handling
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
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            # Fallback to existing method
            ai_response = "I'm sorry, I'm having trouble processing your request right now. Could you please try again?"


        # Save agent response to MongoDB
        if self.conversation_id and self.user_id:
            MongoDB.save_message(
                conversation_id=self.conversation_id,
                user_id=self.user_id,
                sender='agent',
                text=ai_response
            )

        # Store in Pinecone
        store_in_pinecone("brand-designer", "assistant", ai_response)
        
        return ai_response

def get_brand_designer_agent(user_id: str = None, conversation_id: str = None):
    return BrandDesignerAgent(user_id, conversation_id)