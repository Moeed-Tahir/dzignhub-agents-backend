import os
from openai import OpenAI
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool, AgentType
from core.config import OPENAI_API_KEY, PINECONE_API_KEY, ANTHROPIC_API
from core.database import MongoDB
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime
import json
import re

# ---------------------------
# Pinecone Setup
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
openai_client = OpenAI(api_key=OPENAI_API_KEY)  # For embeddings only

# ---------------------------
# Helper Functions
# ---------------------------
def embed_text(text: str):
    """Create an embedding for given text using OpenAI (for search only)."""
    return openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

def store_in_pinecone(agent_type: str, role: str, text: str, user_id: str, conversation_id: str = None):
    """Store embedding with metadata in Pinecone"""
    try:
        vector = embed_text(text)
        
        metadata = {
            "agent_type": agent_type,
            "role": role,
            "conversation_id": conversation_id,
            "user_id": user_id
        }
        
        # Remove None values
        metadata = {k: v for k, v in metadata.items() if v is not None}
        
        vector_id = f"{agent_type}-{role}-{hash(text)}-{datetime.utcnow().timestamp()}"
        
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

def search_content_conversations(query: str, user_id: str, agent_type: str = "content-creator", top_k: int = 10):
    """Search content creation conversations"""
    try:
        print(f"[DEBUG] Searching content conversations for: '{query}' (user: {user_id})")
        
        query_vector = embed_text(query)
        
        search_results = pinecone_index.query(
            vector=query_vector,
            top_k=top_k * 3,
            include_metadata=True,
            include_values=False,
            filter={
                "agent_type": agent_type,
                "user_id": user_id
            }
        )
        
        print(f"[DEBUG] Found {len(search_results.matches)} vector matches")
        
        conversation_scores = {}
        
        for match in search_results.matches:
            if match.score < 0.2:
                continue
                
            metadata = match.metadata or {}
            conv_id = metadata.get('conversation_id')
            
            if not conv_id:
                continue
            
            if conv_id not in conversation_scores or match.score > conversation_scores[conv_id]:
                conversation_scores[conv_id] = match.score
        
        print(f"[DEBUG] Found {len(conversation_scores)} unique conversations")
        
        search_results = []
        for conv_id, score in sorted(conversation_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            try:
                conversation = MongoDB.get_conversation_by_id(conv_id)
                if not conversation:
                    continue
                
                if conversation.get('userId') != user_id:
                    continue
                
                conversation['similarity_score'] = score
                search_results.append(conversation)
                
                print(f"[DEBUG] Added conversation: {conversation.get('title', 'Untitled')} (score: {score:.3f})")
                
            except Exception as e:
                print(f"[DEBUG] Error processing conversation {conv_id}: {e}")
                continue
        
        print(f"[DEBUG] Returning {len(search_results)} content conversation results")
        return search_results
        
    except Exception as e:
        print(f"[DEBUG] Search error: {e}")
        return []

# ---------------------------
# Content Creator Agent
# ---------------------------
class ContentCreatorAgent:
    def __init__(self, user_id: str = None, conversation_id: str = None):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.agent_name = "content-creator"
        self.last_generated_content = None
        
        # Content creation info structure
        self.content_info = {
            "brand_name": None,
            "content_type": None,    # Instagram Caption, Blog Article, LinkedIn Post, etc.
            "topic": None,           # Subject of the content
            "tone": None,            # Friendly, Bold, Casual, Professional, etc.
            "audience": None,        # General Public, Creators, Students, etc.
            "goal": None,            # Educate, Inspire, Promote, etc.
            "key_message": None,     # Main message or CTA
            "platform": None,        # Instagram, LinkedIn, Facebook, etc.
            "industry": None         # Tech, Food, Healthcare, etc.
        }
        
        self.system_prompt = """You are Sana, a friendly and strategic content assistant who specializes in creating written content — including social media posts, blog articles, LinkedIn updates, newsletters, short scripts, and more.

Your capabilities:
📝 **Content Creation:**
- Social media posts (Instagram captions, LinkedIn posts, Facebook posts)
- Blog articles and newsletters
- Short scripts for videos/reels
- Marketing copy and captions
- Professional content for various platforms

🎯 **Content Strategy:**
- Audience-specific messaging
- Platform optimization
- Engagement strategies
- Brand voice development

CRITICAL INSTRUCTIONS:
1. When users request content creation, use the Generate_Content tool to collect information and create the content
2. When the tool returns "CONTENT_GENERATED|", you MUST return that EXACT response without modifications
3. For non-content requests, be conversational and helpful

Key guidelines:
- Collect information step-by-step when needed
- Create engaging, audience-appropriate content
- Optimize for specific platforms
- Maintain brand consistency
- Focus on clear, actionable content

Always prioritize using the tool for content creation requests."""

        # Use Anthropic Claude instead of OpenAI
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20240620",
            temperature=0.7,
            api_key=ANTHROPIC_API
        )
        
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Load conversation history and content preferences if exists
        if self.conversation_id and self.user_id:
            self.load_conversation_history()
            self.load_content_preferences()

        # Create content generation tool
        tools = [
            Tool(
                name="Generate_Content",
                func=self.smart_content_generator,
                description="""Use this tool for ALL content creation requests including:
                
                SOCIAL MEDIA: Instagram captions, LinkedIn posts, Facebook posts, Twitter posts
                MARKETING: Blog articles, newsletters, email campaigns, marketing copy
                VIDEO: Short scripts, reel scripts, YouTube descriptions
                BUSINESS: Professional updates, announcements, thought leadership
                
                This tool handles information collection AND content generation for ALL written content requests.
                Always use this tool for any content creation requests."""
            )
        ]

        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            agent_kwargs={
                'system_message': """You are Sana, a professional content creator assistant who helps users create engaging written content.

Your capabilities:
- Social media content (Instagram, LinkedIn, Facebook, Twitter)
- Blog articles and newsletters
- Marketing copy and campaigns
- Video scripts and descriptions

When users mention wanting ANY written content, immediately use the Generate_Content tool.

CRITICAL INSTRUCTION: When the tool returns a response that starts with "CONTENT_GENERATED|", you MUST return that EXACT response without any modifications, additions, or formatting changes.

Examples:
- If tool returns: "CONTENT_GENERATED|Here's your content...|message"
- You return: "CONTENT_GENERATED|Here's your content...|message" (EXACTLY)

For all other responses, be conversational and friendly.

Key guidelines:
- Always use the tool for ANY content creation requests
- NEVER modify CONTENT_GENERATED responses
- Trust the tool to handle information collection and generation
- Focus on being helpful and creative for non-content conversations

Always prioritize using the tool over giving generic advice."""
            }
        )

    def extract_content_info_from_conversation(self, messages):
        """Extract content creation information from conversation history using Claude"""
        
        conversation_text = ""
        for msg in messages:
            role = "User" if msg['sender'] == 'user' else "Assistant"
            conversation_text += f"{role}: {msg['text']}\n"
        
        if not conversation_text.strip():
            print("[DEBUG] No conversation history to extract from")
            return
        
        print(f"[DEBUG] Extracting content info from conversation: {conversation_text[:500]}...")
        
        extraction_prompt = f"""
        Analyze this conversation and extract content creation information.
        
        Conversation:
        {conversation_text}

        Extract the most recent content requirements:
        - brand_name: Company/brand name mentioned
        - content_type: Instagram Caption, Blog Article, LinkedIn Post, Newsletter, etc.
        - topic: What the content is about
        - tone: Friendly, Bold, Casual, Professional, Playful, etc.
        - audience: General Public, Creators, Students, Founders, Developers, etc.
        - goal: Educate, Inspire, Promote, Raise Awareness, Entertain, etc.
        - key_message: Main message or call-to-action
        - platform: Instagram, LinkedIn, Facebook, Twitter, etc.
        - industry: Tech, Food, Healthcare, Finance, etc.

        Return ONLY valid JSON:
        {{"brand_name": "extracted or null", "content_type": "detected type or null", "topic": "content topic or null", "tone": "detected tone or null", "audience": "target audience or null", "goal": "content goal or null", "key_message": "main message or null", "platform": "platform or null", "industry": "industry or null"}}
        """
        
        try:
            response = self.llm.invoke([
                {"role": "system", "content": "Extract content creation information from conversations. Return ONLY valid JSON."},
                {"role": "user", "content": extraction_prompt}
            ])
            
            extracted_text = response.content.strip()
            print(f"[DEBUG] Content extraction result: {extracted_text}")
            
            # Clean up markdown
            if extracted_text.startswith('```json'):
                extracted_text = extracted_text.replace('```json', '').replace('```', '').strip()
            elif extracted_text.startswith('```'):
                extracted_text = extracted_text.replace('```', '').strip()
            
            extracted_info = json.loads(extracted_text)
            
            # Update content_info with extracted information
            for key, value in extracted_info.items():
                if key in self.content_info and value and value.lower() not in ["null", "", "none"]:
                    self.content_info[key] = value
                    print(f"[DEBUG] Updated {key}: {value}")
            
            print(f"[DEBUG] Final content_info: {self.content_info}")
            
        except Exception as e:
            print(f"[DEBUG] Content extraction error: {e}")
            
            # Manual fallback
            print("[DEBUG] Using manual content info search")
            for msg in reversed(messages):
                text = msg['text'].lower()
                
                # Look for content types
                content_types = {
                    "instagram": "Instagram Caption",
                    "linkedin": "LinkedIn Post", 
                    "facebook": "Facebook Post",
                    "blog": "Blog Article",
                    "newsletter": "Newsletter",
                    "script": "Short Script",
                    "twitter": "Twitter Post",
                    "email": "Email Campaign"
                }
                
                for keyword, content_type in content_types.items():
                    if keyword in text:
                        self.content_info['content_type'] = content_type
                        print(f"[DEBUG] Manual extraction - content_type: {content_type}")
                        break

    def extract_from_current_input(self, user_input: str):
        """Extract content information from current user input using Claude"""
        
        current_info_text = ""
        if any(v for v in self.content_info.values()):
            current_info_text = f"Current content project:\n"
            for key, value in self.content_info.items():
                if value:
                    current_info_text += f"- {key}: {value}\n"
            current_info_text += "\n"
        
        extraction_prompt = f"""
        {current_info_text}User input: "{user_input}"
        
        Extract content creation information and return ONLY valid JSON:
        
        Content type mapping:
        - "instagram post/caption" → "Instagram Caption"
        - "linkedin post/update" → "LinkedIn Post"
        - "facebook post" → "Facebook Post"
        - "blog article/post" → "Blog Article"
        - "newsletter" → "Newsletter"
        - "script" → "Short Script"
        - "twitter post/tweet" → "Twitter Post"
        - "email campaign" → "Email Campaign"
        
        Tone mapping:
        - "friendly" → "Friendly"
        - "professional" → "Professional"
        - "casual" → "Casual"
        - "bold" → "Bold"
        - "playful" → "Playful"
        
        Return ONLY JSON - no explanations:
        {{"brand_name": "extracted or null", "content_type": "mapped type or null", "topic": "content topic or null", "tone": "mapped tone or null", "audience": "target audience or null", "goal": "content goal or null", "key_message": "main message or null", "platform": "platform or null"}}
        """
        
        try:
            response = self.llm.invoke([
                {"role": "system", "content": "Extract content creation info. Return ONLY valid JSON."},
                {"role": "user", "content": extraction_prompt}
            ])
            
            extracted_text = response.content.strip()
            print(f"[DEBUG] Current input extraction: {extracted_text}")
            
            # Clean up response
            if "```json" in extracted_text:
                extracted_text = extracted_text.split("```json")[1].split("```")[0].strip()
            elif "```" in extracted_text:
                extracted_text = extracted_text.split("```")[1].split("```")[0].strip()
            
            # Find JSON object
            json_match = re.search(r'\{[^}]+\}', extracted_text)
            if json_match:
                extracted_text = json_match.group(0)
            
            extracted_info = json.loads(extracted_text)
            
            # Update content_info with new information
            for key, value in extracted_info.items():
                if key in self.content_info and value and value.lower() not in ["null", "", "none"]:
                    self.content_info[key] = value
                    print(f"[DEBUG] Updated from current input - {key}: {value}")
            
            print(f"[DEBUG] Final content_info after current input: {self.content_info}")
                    
        except Exception as e:
            print(f"[DEBUG] Current input extraction error: {e}")
            
            # Simple fallback
            print("[DEBUG] Using simple keyword fallback")
            user_lower = user_input.lower()
            
            # Detect content types
            if "instagram" in user_lower:
                self.content_info['content_type'] = "Instagram Caption"
                self.content_info['platform'] = "Instagram"
            elif "linkedin" in user_lower:
                self.content_info['content_type'] = "LinkedIn Post"
                self.content_info['platform'] = "LinkedIn"
            elif "facebook" in user_lower:
                self.content_info['content_type'] = "Facebook Post"
                self.content_info['platform'] = "Facebook"
            elif "blog" in user_lower:
                self.content_info['content_type'] = "Blog Article"
            elif "newsletter" in user_lower:
                self.content_info['content_type'] = "Newsletter"
            elif "script" in user_lower:
                self.content_info['content_type'] = "Short Script"
            elif "twitter" in user_lower or "tweet" in user_lower:
                self.content_info['content_type'] = "Twitter Post"
                self.content_info['platform'] = "Twitter"
            
            print(f"[DEBUG] Fallback extracted content type: {self.content_info.get('content_type')}")

    def load_content_preferences(self):
        """Load content preferences from User.contentPreferences field"""
        if not self.user_id:
            return
            
        try:
            content_prefs = MongoDB.get_user_content_preferences(self.user_id)
            
            # Update content_info with saved preferences
            for key, value in content_prefs.items():
                if key in self.content_info and value:
                    self.content_info[key] = value
                    print(f"[DEBUG] Loaded content preference {key}: {value}")
            
            print(f"[DEBUG] Loaded content preferences: {self.content_info}")
                
        except Exception as e:
            print(f"[DEBUG] Error loading content preferences: {e}")

    def save_content_preferences(self):
        """Save current content_info to User.contentPreferences field"""
        if not self.user_id:
            return
            
        try:
            # Filter out None values and add timestamp
            content_prefs_data = {k: v for k, v in self.content_info.items() if v is not None}
            content_prefs_data["lastUpdated"] = datetime.utcnow().isoformat()
            
            success = MongoDB.update_user_content_preferences(self.user_id, content_prefs_data)
            
            if success:
                print(f"[DEBUG] Saved content preferences: {content_prefs_data}")
            else:
                print("[DEBUG] Failed to save content preferences")
                
        except Exception as e:
            print(f"[DEBUG] Error saving content preferences: {e}")

    def detect_content_intent(self, user_input: str) -> bool:
        """Detect if user wants to create content using Claude"""
        
        intent_prompt = f"""
        Analyze this user message and determine if they want to create written content NOW.
        
        User message: "{user_input}"
        
        Return ONLY "YES" if they want to create content now, or "NO" if they're just providing information.
        
        Examples that mean YES (create content):
        - "write an Instagram post" → YES
        - "create LinkedIn post" → YES  
        - "generate blog article" → YES
        - "write newsletter" → YES
        - "create script" → YES
        - "write content for..." → YES
        - "generate post about..." → YES
        
        Examples that mean NO (just providing info):
        - "my brand name is PsychoDevs" → NO
        - "I want casual tone" → NO
        - "target audience is developers" → NO
        
        Answer: """
        
        try:
            response = self.llm.invoke([
                {"role": "system", "content": "Detect if user wants to create content NOW. Always respond with only YES or NO."},
                {"role": "user", "content": intent_prompt}
            ])
            
            intent = response.content.strip().upper()
            print(f"[DEBUG] Content creation intent detected: {intent} for input: '{user_input}'")
            
            return intent == "YES"
            
        except Exception as e:
            print(f"[DEBUG] Intent detection error: {e}")
            # Enhanced fallback keywords for content creation
            creation_keywords = [
                "write", "create", "generate", "make", "compose",
                "post", "caption", "article", "newsletter", "script",
                "instagram", "linkedin", "facebook", "twitter", "blog"
            ]
            
            return any(phrase in user_input.lower() for phrase in creation_keywords)

    def smart_content_generator(self, user_input: str = "") -> str:
        """Smart content generation with automatic information collection"""
        
        print(f"[DEBUG] smart_content_generator called with input: '{user_input}'")
        print(f"[DEBUG] Initial content_info: {self.content_info}")
        
        self.user_context = user_input
        
        # Process input
        processed_input = user_input
        if user_input.startswith('{"') and user_input.endswith('}'):
            try:
                parsed_input = json.loads(user_input)
                processed_input = parsed_input.get("description", "create content")
            except:
                pass
        
        # Get conversation messages
        recent_messages = []
        if self.conversation_id and self.user_id:
            messages = MongoDB.get_conversation_messages(self.conversation_id, self.user_id)
            recent_messages = messages
            print(f"[DEBUG] Found {len(recent_messages)} total messages for analysis")
        
        # Extract information from conversation
        if recent_messages:
            self.extract_content_info_from_conversation(recent_messages)
        
        # Extract from current input
        if processed_input and processed_input.strip():
            self.extract_from_current_input(processed_input)
        
        # Save updated content_info
        self.save_content_preferences()
        
        print(f"[DEBUG] Current content_info after extraction: {self.content_info}")
        
        # Check missing information
        missing_info = [k for k, v in self.content_info.items() if not v]
        provided_info = {k: v for k, v in self.content_info.items() if v}
        
        print(f"[DEBUG] Missing info: {missing_info}")
        print(f"[DEBUG] Provided info: {provided_info}")
        
        # Check if we have minimum info to generate content
        has_minimum_info = (
            self.content_info.get("content_type") and 
            self.content_info.get("topic")
        )
        
        # Detect creation intent
        wants_creation = self.detect_content_intent(user_input) if user_input else False
        
        # Generate content if requested and we have minimum info
        if wants_creation and has_minimum_info:
            # Auto-complete missing fields
            if missing_info:
                print("[DEBUG] Auto-completing missing fields for content generation...")
                auto_completed = self.intelligent_auto_complete(self.content_info.copy())
                for key, value in auto_completed.items():
                    if not self.content_info.get(key):
                        self.content_info[key] = value
                self.save_content_preferences()
            
            print(f"[DEBUG] Generating content...")
            content_result = self.generate_content_with_claude(self.content_info, user_context=user_input)
            
            if content_result["type"] == "content_generated":
                self.last_generated_content = content_result["content"]
                return f"""CONTENT_GENERATED|{content_result['content']}|{content_result['message']}"""
            else:
                return content_result["message"]
        
        # Ask for missing information if not enough to generate
        if not self.content_info.get("content_type"):
            return self.ask_for_content_type()
        elif not self.content_info.get("topic"):
            return self.ask_for_topic()
        else:
            # Use natural conversation collection
            return self.collect_content_info(missing_info, provided_info)

    def intelligent_auto_complete(self, provided_info: dict):
        """Auto-complete missing content information using Claude"""
        
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
        Based on the following content information, intelligently suggest appropriate values for the missing fields.
        
        Known information:
        {known_info}
        
        Please complete this JSON with intelligent defaults for the missing fields: {missing_fields}
        
        Guidelines:
        - For tone: Consider the content type and audience
        - For audience: Match the platform and content type
        - For goal: Align with content type purpose
        - For platform: Match the content type
        - For industry: Use context clues from brand/topic
        
        Return ONLY a complete JSON object with all fields:
        {{"brand_name": "value", "content_type": "value", "topic": "value", "tone": "value", "audience": "value", "goal": "value", "key_message": "value", "platform": "value", "industry": "value"}}
        """
        
        try:
            response = self.llm.invoke([
                {"role": "system", "content": "Complete content information with intelligent defaults. Return ONLY valid JSON."},
                {"role": "user", "content": completion_prompt}
            ])
            
            completed_text = response.content.strip()
            print(f"[DEBUG] Auto-completion result: {completed_text}")
            
            # Clean up markdown
            if completed_text.startswith('```json'):
                completed_text = completed_text.replace('```json', '').replace('```', '').strip()
            elif completed_text.startswith('```'):
                completed_text = completed_text.replace('```', '').strip()
            
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

    def generate_content_with_claude(self, info: dict, user_context: str = ""):
        """Generate content using Claude with dynamic prompts"""
        
        content_type = info.get('content_type', 'Social Media Post')
        topic = info.get('topic', 'General topic')
        tone = info.get('tone', 'Friendly')
        audience = info.get('audience', 'General audience')
        goal = info.get('goal', 'Engage')
        brand_name = info.get('brand_name', '')
        key_message = info.get('key_message', '')
        platform = info.get('platform', '')
        industry = info.get('industry', '')
        
        # Build dynamic prompt based on content type
        content_prompts = {
            "Instagram Caption": f"Create an engaging Instagram caption about {topic}. Tone: {tone}. Audience: {audience}. Goal: {goal}. Include relevant hashtags and emojis. Keep it authentic and scroll-stopping.",
            
            "LinkedIn Post": f"Write a professional LinkedIn post about {topic}. Tone: {tone}. Target: {audience}. Purpose: {goal}. Include industry insights and encourage engagement. Professional yet personable.",
            
            "Facebook Post": f"Create a Facebook post about {topic}. Tone: {tone}. Audience: {audience}. Goal: {goal}. Make it conversational and community-focused. Encourage comments and shares.",
            
            "Blog Article": f"Write a comprehensive blog article about {topic}. Tone: {tone}. Target readers: {audience}. Purpose: {goal}. Include introduction, main points, and conclusion. Make it informative and engaging.",
            
            "Newsletter": f"Create a newsletter section about {topic}. Tone: {tone}. Subscribers: {audience}. Goal: {goal}. Make it valuable and actionable for your subscribers.",
            
            "Short Script": f"Write a short script about {topic}. Tone: {tone}. Viewers: {audience}. Purpose: {goal}. Keep it under 60 seconds, engaging, and action-oriented.",
            
            "Twitter Post": f"Create a Twitter post about {topic}. Tone: {tone}. Audience: {audience}. Goal: {goal}. Keep it under 280 characters, impactful, and shareable.",
            
            "Email Campaign": f"Write an email campaign about {topic}. Tone: {tone}. Recipients: {audience}. Purpose: {goal}. Include subject line and compelling body text."
        }
        
        base_prompt = content_prompts.get(content_type, f"Create {content_type.lower()} content about {topic}.")
        
        # Add context if provided
        context_section = ""
        if user_context and user_context.strip():
            context_section = f" Additional context: {user_context}"
        
        # Add brand and message info
        brand_section = ""
        if brand_name:
            brand_section = f" Brand: {brand_name}."
        if key_message:
            brand_section += f" Key message: {key_message}."
        
        # Build complete prompt
        full_prompt = f"{base_prompt}{context_section}{brand_section}"
        
        if platform:
            full_prompt += f" Optimize for {platform}."
        if industry:
            full_prompt += f" Industry context: {industry}."
        
        full_prompt += " Make it authentic, engaging, and aligned with the specified tone and goals."
        
        try:
            print(f"[DEBUG] Generating {content_type} with Claude...")
            print(f"[DEBUG] Topic: {topic}")
            print(f"[DEBUG] Tone: {tone}")
            print(f"[DEBUG] Audience: {audience}")
            print(f"[DEBUG] Goal: {goal}")
            print(f"[DEBUG] Prompt: {full_prompt}")
            
            response = self.llm.invoke([
                {"role": "system", "content": f"You are an expert content creator specializing in {content_type}. Create engaging, authentic content that resonates with the target audience."},
                {"role": "user", "content": full_prompt}
            ])
            
            generated_content = response.content.strip()
            print(f"[DEBUG] Successfully generated {content_type}!")
            
            # Create success message
            success_message = f"🎯 **Your {content_type} is ready!**\n\nI've created {tone.lower()} content about {topic} tailored for {audience}. The content is optimized to {goal.lower()} and matches your brand voice."
            
            # Add platform-specific tips
            platform_tips = {
                "Instagram Caption": "\n\n💡 **Tip**: Post during peak hours (6-9 PM) for maximum engagement!",
                "LinkedIn Post": "\n\n💡 **Tip**: Post on Tuesday-Thursday mornings for best professional reach!",
                "Facebook Post": "\n\n💡 **Tip**: Encourage comments by asking questions to boost engagement!",
                "Twitter Post": "\n\n💡 **Tip**: Tweet during lunch hours or evening commute for better visibility!"
            }
            
            if content_type in platform_tips:
                success_message += platform_tips[content_type]
            
            return {
                "type": "content_generated",
                "message": success_message,
                "content": generated_content,
                "content_type": content_type,
                "content_info": info
            }
            
        except Exception as e:
            print(f"[DEBUG] Claude generation error: {str(e)}")
            return {
                "type": "error",
                "message": f"I encountered an issue generating your {content_type}: {str(e)}. Let me try again with different specifications.",
                "content": None,
                "content_info": info
            }

    def ask_for_content_type(self) -> str:
        """Ask user for content type"""
        return """🎯 **What type of content would you like to create?**

Choose from these popular options:

📱 **Social Media:**
• Instagram Caption
• LinkedIn Post  
• Facebook Post
• Twitter Post

📝 **Long-form Content:**
• Blog Article
• Newsletter
• Email Campaign

🎬 **Video Content:**
• Short Script (for reels/videos)

Just let me know which one you'd prefer, or tell me about your specific content needs!"""

    def ask_for_topic(self) -> str:
        """Ask user for content topic"""
        content_type = self.content_info.get('content_type', 'content')
        return f"""Great choice! Now, what topic would you like your **{content_type}** to be about?

For example:
• Product launch announcement
• Company milestone celebration  
• Industry insights or tips
• Behind-the-scenes content
• Customer success story
• Educational content

Just describe what you want to write about, and I'll help you create engaging content around that topic! 💭"""

    def collect_content_info(self, missing_info: list, provided_info: dict) -> str:
        """Generate natural conversation responses for collecting missing info"""
        
        # If we have content type and topic, we can generate with smart defaults
        if provided_info.get("content_type") and provided_info.get("topic"):
            print("[DEBUG] Have minimum info, auto-completing and generating...")
            auto_completed = self.intelligent_auto_complete(self.content_info.copy())
            for key, value in auto_completed.items():
                if not self.content_info.get(key):
                    self.content_info[key] = value
            self.save_content_preferences()
            
            # Generate the content
            content_result = self.generate_content_with_claude(self.content_info)
            
            if content_result["type"] == "content_generated":
                self.last_generated_content = content_result["content"]
                return f"""CONTENT_GENERATED|{content_result['content']}|{content_result['message']}"""
            else:
                return content_result["message"]
        
        # Ask for next missing field naturally
        content_type = provided_info.get('content_type', 'content')
        
        if 'tone' in missing_info:
            return f"""Perfect! For your **{content_type}**, what tone would you like?

🎨 **Tone Options:**
• **Friendly** - Warm and approachable
• **Professional** - Formal and authoritative  
• **Casual** - Relaxed and conversational
• **Bold** - Confident and attention-grabbing
• **Playful** - Fun and energetic

What vibe fits your brand best? 😊"""
        
        elif 'audience' in missing_info:
            return f"""Great! Who is your target audience for this **{content_type}**?

👥 **Audience Examples:**
• General Public
• Developers/Tech professionals
• Business owners/Founders
• Students
• Creators/Designers
• Industry professionals

Who are you trying to reach? 🎯"""
        
        elif 'goal' in missing_info:
            return f"""Excellent! What's the main goal of this **{content_type}**?

🎯 **Content Goals:**
• **Educate** - Share knowledge/tips
• **Inspire** - Motivate your audience
• **Promote** - Showcase product/service
• **Raise Awareness** - Build brand recognition
• **Entertain** - Engage and delight

What do you want to achieve? ✨"""
        
        else:
            # Generate with what we have
            return "I have enough information to create great content for you! Let me generate it now. 🚀"

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
        """Handle user query using LangChain agent with Claude"""
        
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
            agent_type="content-creator", 
            role="user", 
            text=query,
            user_id=self.user_id,
            conversation_id=self.conversation_id
        )

        # Retrieve similar past queries
        past_results = retrieve_from_pinecone(query)
        if past_results.matches:
            print(f"[DEBUG] Similar past entries found: {past_results.matches}")

        try:
            # Use LangChain agent with Claude
            print(f"[DEBUG] Using LangChain agent with Claude")
            ai_response = self.agent.run(query)
            
            # Check if LangChain corrupted a CONTENT_GENERATED response
            if self.last_generated_content and "CONTENT_GENERATED|" not in ai_response:
                print(f"[DEBUG] LangChain corrupted the CONTENT_GENERATED response")
                print(f"[DEBUG] Original response: {ai_response}")
                
                # Reconstruct the proper format
                if self.last_generated_content in ai_response:
                    success_message = "I've created engaging content that perfectly matches your requirements."
                    ai_response = f"CONTENT_GENERATED|{self.last_generated_content}|{success_message}"
                    print(f"[DEBUG] Reconstructed proper format: {ai_response}")
            
        except Exception as e:
            print(f"LangChain agent error: {e}, falling back to Claude direct")
            
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

            # Fallback to Claude direct
            try:
                response = self.llm.invoke([
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": enhanced_query}
                ])
                
                ai_response = response.content
                print(f"[DEBUG] Using Claude direct approach (fallback)")
                
            except Exception as e2:
                print(f"Both LangChain and Claude failed: {e2}")
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
            agent_type="content-creator", 
            role="agent", 
            text=ai_response,
            user_id=self.user_id,
            conversation_id=self.conversation_id
        )
        
        return ai_response

def get_content_creator_agent(user_id: str = None, conversation_id: str = None):
    return ContentCreatorAgent(user_id, conversation_id)