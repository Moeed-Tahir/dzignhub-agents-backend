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

        self.design_info = {
            "brand_name": None,
            "logo_type": None,
            "target_audience": None,
            "color_palette": None
        }


        # Add the system prompt as an attribute
        self.system_prompt = """You are Zara, a professional brand designer assistant who helps users create logos and develop brand identities. Give response in markdown format.

Your capabilities:
- Logo design and generation using DALL-E
- Color palette recommendations
- Typography suggestions
- Brand identity consultation

When users mention wanting a logo, creating a logo, designing a brand, or anything related to logo generation, you should help them by collecting the necessary information first.

You need to gather:
1. Brand name (required)
2. Logo type (text-based, icon-based, mascot, combination, etc.)
3. Target audience 
4. Color preferences

Once you have sufficient information, you can generate the logo.

Key guidelines:
- Be conversational and friendly
- Ask clarifying questions when needed
- Help users think through their brand identity
- Focus on being helpful and creative

Always prioritize understanding the user's needs before making suggestions."""
     
        
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
        name="Generate_Logo_with_DALLE",
        func=self.smart_logo_generator,
        description="""Use this tool IMMEDIATELY when the user:
        - Asks to generate/create/design/make a logo
        - Says "give me the logo" or "show me the logo" 
        - Requests visual branding materials
        - Mentions DALL-E or image generation
        - Asks to see their brand design
        
        This tool handles information collection AND actual logo generation.
        Always use this tool for any logo-related requests, even if you think you don't have enough info."""
    )
]

        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
              agent_kwargs={
        'system_message': """You are Zara, a professional brand designer assistant who helps users create logos and develop brand identities.

Your capabilities:
- Logo design and generation using DALL-E
- Color palette recommendations
- Typography suggestions
- Brand identity consultation

When users mention wanting a logo, creating a logo, designing a brand, or anything related to logo generation, immediately use the Generate_Logo_with_DALLE tool.

CRITICAL INSTRUCTION: When the tool returns a response that starts with "LOGO_GENERATED|", you MUST return that EXACT response without any modifications, additions, or formatting changes. Do not convert it to markdown, do not add explanations, just return the exact string as-is.

Examples:
- If tool returns: "LOGO_GENERATED|https://image-url|message"
- You return: "LOGO_GENERATED|https://image-url|message" (EXACTLY)

For all other responses, be conversational and friendly.

Key guidelines:
- Always use the tool for logo-related requests
- NEVER modify LOGO_GENERATED responses
- Trust the tool to handle information collection and generation
- Focus on being helpful and creative for non-logo conversations

Always prioritize using the tool over giving generic advice."""
    }
    )

    def extract_brand_info_from_conversation(self, messages):
        """Use GPT to intelligently extract brand information from conversation"""
        
        # Build conversation text
        conversation_text = ""
        for msg in messages:
            role = "User" if msg['sender'] == 'user' else "Assistant"
            conversation_text += f"{role}: {msg['text']}\n"
        
        if not conversation_text.strip():
            print("[DEBUG] No conversation history to extract from")
            return
        
        print(f"[DEBUG] Extracting from conversation: {conversation_text}")
        
        # IMPORTANT: Include current design_info in the prompt so GPT knows what we already have
        current_info_text = ""
        if any(v for v in self.design_info.values()):
            current_info_text = f"\nCurrently known information:\n"
            for key, value in self.design_info.items():
                if value:
                    current_info_text += f"- {key}: {value}\n"
        
        # Use GPT to extract brand information
        extraction_prompt = f"""
        Analyze the following conversation and extract brand information. Return ONLY a JSON object with the extracted information.
        
        IMPORTANT: If information is already known (shown below), include it in your response to preserve it.
        {current_info_text}

        Conversation:
        {conversation_text}

        Extract these fields and PRESERVE any existing values:
        - brand_name: The name of the brand/business/company
        - logo_type: The style of logo (text-based, icon-based, mascot, combination, minimalist, modern, etc.)
        - target_audience: Who the brand is targeting (families, professionals, young people, etc.)
        - color_palette: Color preferences mentioned (specific colors, moods like "professional", "vibrant", etc.)

        Return format (preserve existing values, add new ones):
        {{
            "brand_name": "extracted or existing name",
            "logo_type": "extracted or existing type", 
            "target_audience": "extracted or existing audience",
            "color_palette": "extracted or existing colors"
        }}
        """
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at extracting structured information from conversations. Preserve existing information and add new details. Always return valid JSON."},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=200
            )
            
            extracted_text = response.choices[0].message.content.strip()
            print(f"[DEBUG] GPT extraction result: {extracted_text}")
            
            # Parse the JSON response
            import json
            extracted_info = json.loads(extracted_text)
            
            # Update design_info with extracted information (now preserves existing values)
            for key, value in extracted_info.items():
                if value and value.lower() != "null":
                    self.design_info[key] = value
                    print(f"[DEBUG] Set {key}: {value}")
            
            print(f"[DEBUG] Updated design_info: {self.design_info}")
            
        except json.JSONDecodeError as e:
            print(f"[DEBUG] JSON parsing error: {e}")
            print(f"[DEBUG] Raw response: {extracted_text}")
        except Exception as e:
            print(f"[DEBUG] Extraction error: {e}")

    def intelligent_auto_complete(self, provided_info: dict):
        """Use GPT to intelligently fill missing brand information based on what's provided"""
        
        # Build context from what we know
        known_info = ""
        missing_fields = []
        for key, value in provided_info.items():
            if value:
                known_info += f"{key}: {value}\n"
            else:
                missing_fields.append(key)
        
        if not known_info.strip():
            return provided_info  # Nothing to work with
        
        completion_prompt = f"""
        Based on the following brand information, intelligently suggest appropriate values for the missing fields.
        Make educated guesses based on industry standards and the brand context.

        Known information:
        {known_info}

        Please complete this JSON with intelligent defaults for the missing fields: {missing_fields}

        Guidelines:
        - For logo_type: Choose based on brand name and industry (tech = modern/minimalist, food = combination, etc.)
        - For target_audience: Infer from brand name and type (TechStart = young professionals, KidsPlay = families, etc.)  
        - For color_palette: Suggest colors that match the industry and target audience

        Return ONLY a complete JSON object:
        {{
            "brand_name": "{provided_info.get('brand_name', 'Modern Brand')}",
            "logo_type": "suggest based on brand context",
            "target_audience": "suggest based on brand context", 
            "color_palette": "suggest based on brand context"
        }}

        Examples:
        - PsychoDevs (software agency) → logo_type: "modern minimalist", target_audience: "tech companies and startups", color_palette: "professional blues and grays"
        - Sweet Dreams Bakery → logo_type: "combination", target_audience: "families and dessert lovers", color_palette: "warm pastels like pink, cream, and gold"
        """
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert brand consultant. Make intelligent suggestions based on brand context. Always return valid JSON."},
                    {"role": "user", "content": completion_prompt}
                ],
                temperature=0.3,  # Lower temperature for consistent suggestions
                max_tokens=300
            )
            
            completed_text = response.choices[0].message.content.strip()
            print(f"[DEBUG] Auto-completion result: {completed_text}")
            
            import json
            completed_info = json.loads(completed_text)
            
            # Merge with original info (original takes precedence)
            final_info = {}
            for key in provided_info.keys():
                final_info[key] = provided_info[key] if provided_info[key] else completed_info.get(key)
            
            print(f"[DEBUG] Final completed info: {final_info}")
            return final_info
            
        except Exception as e:
            print(f"[DEBUG] Auto-completion error: {e}")
            return provided_info  # Return original if completion fails

    def extract_from_current_input(self, user_input: str):
        """Use GPT to extract brand info from the current user message while preserving existing info"""
        
        # Include current known information in the prompt
        current_info_text = ""
        if any(v for v in self.design_info.values()):
            current_info_text = f"Currently known information:\n"
            for key, value in self.design_info.items():
                if value:
                    current_info_text += f"- {key}: {value}\n"
            current_info_text += "\n"
        
        extraction_prompt = f"""
        {current_info_text}The user just said: "{user_input}"
        
        This might contain additional brand information. Extract any NEW details and PRESERVE existing information.
        
        Return a complete JSON object with all information (existing + new):
        
        Examples:
        - If we already know brand_name="PsychoDevs" and user says "icon-based":
        → {{"brand_name": "PsychoDevs", "logo_type": "icon-based", "target_audience": null, "color_palette": null}}
        
        - If we know nothing and user says "PsychoDevs":
        → {{"brand_name": "PsychoDevs", "logo_type": null, "target_audience": null, "color_palette": null}}
        
        Return format (preserve ALL existing values, add new ones):
        {{"brand_name": "preserve existing or add new", "logo_type": "preserve existing or add new", "target_audience": "preserve existing or add new", "color_palette": "preserve existing or add new"}}
        """
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Preserve existing brand information and add new details from user input. Always return complete JSON with all fields."},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.1,
                max_tokens=150
            )
            
            extracted_text = response.choices[0].message.content.strip()
            print(f"[DEBUG] Current input extraction: {extracted_text}")
            
            import json
            extracted_info = json.loads(extracted_text)
            
            # Update design_info with the complete extracted information
            for key, value in extracted_info.items():
                if value and value.lower() != "null":
                    self.design_info[key] = value
                    print(f"[DEBUG] Updated from current input - {key}: {value}")
            
            print(f"[DEBUG] Final design_info after current input: {self.design_info}")
                    
        except Exception as e:
            print(f"[DEBUG] Current input extraction error: {e}")


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
        """Use GPT to detect if user wants to generate a logo"""
        
        intent_prompt = f"""
        Analyze this user message and determine if they want to generate/create a logo NOW.
        
        User message: "{user_input}"
        
        Return ONLY "YES" if they want to generate a logo now, or "NO" if they're just providing information or asking questions.
        
        Examples:
        - "generate now" → YES
        - "create the logo" → YES  
        - "make my logo" → YES
        - "show me the logo" → YES
        - "generate the logo now" → YES
        - "I want to see my logo" → YES
        - "create it" → YES
        - "make it happen" → YES
        - "let's do it" → YES
        - "go for it" → YES
        
        - "my brand name is PsychoDevs" → NO
        - "I want modern style" → NO
        - "what colors work best?" → NO
        - "I need help with my logo" → NO (just asking for help, not ready to generate)
        
        Answer: """
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at detecting user intent. Always respond with only YES or NO."},
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
            # Fallback to simple keyword check
            return any(phrase in user_input.lower() for phrase in ["generate", "create", "make", "show", "build"])


    def smart_logo_generator(self, user_input: str = "") -> str:
        """Smart logo generation with GPT-powered information extraction and auto-completion"""
        
        print(f"[DEBUG] smart_logo_generator called with input: '{user_input}'")
        print(f"[DEBUG] Initial design_info: {self.design_info}")
        
        # Clean up user input if it's JSON from LangChain
        if user_input.startswith('{"') and user_input.endswith('}'):
            try:
                import json
                parsed_input = json.loads(user_input)
                user_input = parsed_input.get("description", "generate logo")
            except:
                pass
        
        # Load fresh design_info from database if it's empty
        if not any(v for v in self.design_info.values()):
            print("[DEBUG] Design_info empty, reloading from database...")
            self.load_brand_design()
        
        # Get recent conversation context
        recent_messages = []
        if self.conversation_id and self.user_id:
            messages = MongoDB.get_conversation_messages(self.conversation_id, self.user_id)
            recent_messages = messages[-5:]  # Last 5 messages
            print(f"[DEBUG] Found {len(recent_messages)} recent messages")
        
        # Extract information from conversation
        if recent_messages:
            self.extract_brand_info_from_conversation(recent_messages)
        
        # Extract from current user input
        if user_input and user_input.strip():
            self.extract_from_current_input(user_input)
        
        # Save updated design_info to database after extraction
        self.save_brand_design()
        
        print(f"[DEBUG] Current design_info after extraction: {self.design_info}")
        
        # Check what information we're missing
        missing_info = [k for k, v in self.design_info.items() if not v]
        provided_info = {k: v for k, v in self.design_info.items() if v}
        
        print(f"[DEBUG] Missing info: {missing_info}")
        print(f"[DEBUG] Provided info: {provided_info}")
        
        # Use GPT to detect if user wants to generate logo NOW
        wants_generation = self.detect_generation_intent(user_input) if user_input else False
        
        # When user wants to generate with auto-completion
        if wants_generation and provided_info.get("brand_name"):
            # User wants to generate - auto-complete missing fields if needed
            if missing_info:
                print("[DEBUG] Auto-completing missing fields for generation...")
                auto_completed = self.intelligent_auto_complete(self.design_info.copy())
                self.design_info.update(auto_completed)
                self.save_brand_design()  # Save completed info
            
            print("[DEBUG] Generating logo...")
            logo_result = generate_logo_dalle(self.design_info)
            
            if logo_result["type"] == "logo_generated":
                self.last_generated_image = logo_result["image_url"]
                # Return in EXACT format: LOGO_GENERATED|image_url|message
                return f"""LOGO_GENERATED|{logo_result['image_url']}|{logo_result['message']}"""
            else:
                return logo_result["message"]

        # When all info is complete
        if not missing_info:
            print("[DEBUG] All info collected, generating logo...")
            logo_result = generate_logo_dalle(self.design_info)
            
            if logo_result["type"] == "logo_generated":
                self.last_generated_image = logo_result["image_url"]
                # Return in EXACT format: LOGO_GENERATED|image_url|message
                return f"""LOGO_GENERATED|{logo_result['image_url']}|{logo_result['message']}"""
            else:
                return logo_result["message"]

        # Fallback generation
        else:
            # Fallback generation - if we somehow get here with complete info
            print("[DEBUG] Fallback logo generation...")
            if self.design_info.get("brand_name"):
                auto_completed = self.intelligent_auto_complete(self.design_info.copy())
                self.design_info.update(auto_completed)
                self.save_brand_design()
                
                logo_result = generate_logo_dalle(self.design_info)
                if logo_result["type"] == "logo_generated":
                    self.last_generated_image = logo_result["image_url"]
                    # Return in EXACT format: LOGO_GENERATED|image_url|message
                    return f"""LOGO_GENERATED|{logo_result['image_url']}|{logo_result['message']}"""
                else:
                    return logo_result["message"]
            else:
                return "I'd love to create a logo for you! What's the name of your brand or business?" 

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
            if self.last_generated_image and "LOGO_GENERATED|" not in ai_response:
                print(f"[DEBUG] LangChain corrupted the LOGO_GENERATED response")
                print(f"[DEBUG] Original response: {ai_response}")
                
                # Extract the image URL from the corrupted response
                if self.last_generated_image in ai_response or "[Logo](" in ai_response:
                    # Reconstruct the proper format
                    success_message = f"🎉 **Your {self.design_info.get('brand_name', 'brand')} logo is ready!**\n\nI've created a {self.design_info.get('logo_type', 'professional')} design that perfectly captures your brand identity for {self.design_info.get('target_audience', 'your audience')}. The design uses {self.design_info.get('color_palette', 'professional colors')} to create a professional and memorable look."
                    
                    ai_response = f"LOGO_GENERATED|{self.last_generated_image}|{success_message}"
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