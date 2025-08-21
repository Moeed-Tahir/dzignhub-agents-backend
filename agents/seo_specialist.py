import os
from openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool, AgentType
from core.config import OPENAI_API_KEY, PINECONE_API_KEY, GOOGLE_API_KEY
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

def search_seo_conversations(query: str, user_id: str, agent_type: str = "seo-specialist", top_k: int = 10):
    """Search SEO conversations"""
    try:
        print(f"[DEBUG] Searching SEO conversations for: '{query}' (user: {user_id})")
        
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
        
        print(f"[DEBUG] Returning {len(search_results)} SEO conversation results")
        return search_results
        
    except Exception as e:
        print(f"[DEBUG] Search error: {e}")
        return []

# ---------------------------
# SEO Specialist Agent
# ---------------------------
class SEOSpecialistAgent:
    def __init__(self, user_id: str = None, conversation_id: str = None):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.agent_name = "seo-specialist"
        self.last_generated_brief = None
        
        # SEO info structure (based on novi-seo-agent.js)
        self.seo_info = {
            "content_type": None,     # Blog Article, Landing Page, Service Page, etc.
            "topic": None,            # Main topic/theme
            "audience": None,         # Freelancers, Developers, Designers, etc.
            "goal": None,             # Increase Traffic, Improve Ranking, etc.
            "tone": None,             # Professional, Friendly, Bold, etc.
            "primary_keyword": None,  # Main keyword/phrase
            "target_location": None,  # Local SEO (optional)
            "competitor_sites": None  # Competitor analysis (optional)
        }
        
        self.system_prompt = """You are Novi, a smart SEO specialist assistant who creates comprehensive SEO content briefs and strategies.

Your capabilities:
🔍 **SEO Content Briefs:**
- Blog articles and landing pages
- Service and product pages
- Local SEO optimization
- Keyword research and strategy

📊 **SEO Analysis:**
- Content optimization recommendations
- Meta tag suggestions
- Header structure guidance
- Internal linking strategies

🎯 **SEO Strategy:**
- Target audience analysis
- Competitor research insights
- Goal-oriented SEO planning
- Content calendar suggestions

CRITICAL INSTRUCTIONS:
1. When users request SEO help, use the Generate_SEO_Brief tool to collect information and create comprehensive briefs
2. When the tool returns "SEO_BRIEF_GENERATED|", you MUST return that EXACT response without modifications
3. For non-SEO requests, be conversational and helpful

Key guidelines:
- Collect SEO information step-by-step when needed
- Create actionable, data-driven SEO briefs
- Focus on practical implementation
- Provide keyword and meta tag suggestions
- Always consider user intent and search behavior

Always prioritize using the tool for SEO-related requests."""

        # Use Google Gemini instead of Claude/OpenAI
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-latest",
            temperature=0.7,
            google_api_key=GOOGLE_API_KEY
        )
        
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Load conversation history and SEO preferences if exists
        if self.conversation_id and self.user_id:
            self.load_conversation_history()
            self.load_seo_preferences()

        # Create SEO generation tool
        tools = [
            Tool(
                name="Generate_SEO_Brief",
                func=self.smart_seo_generator,
                description="""Use this tool for ALL SEO-related requests including:
                
                CONTENT BRIEFS: Blog articles, landing pages, service pages, product descriptions
                KEYWORD RESEARCH: Primary and secondary keyword suggestions
                META OPTIMIZATION: Title tags, meta descriptions, header structures
                SEO STRATEGY: Content planning, competitor analysis, local SEO
                
                This tool handles information collection AND SEO brief generation for ALL SEO requests.
                Always use this tool for any SEO-related requests."""
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
                'system_message': """You are Novi, a professional SEO specialist assistant who helps users create comprehensive SEO briefs and strategies.

Your capabilities:
- SEO content briefs and optimization
- Keyword research and strategy
- Meta tag optimization
- Content structure recommendations

When users mention wanting ANY SEO help, immediately use the Generate_SEO_Brief tool.

CRITICAL INSTRUCTION: When the tool returns a response that starts with "SEO_BRIEF_GENERATED|", you MUST return that EXACT response without any modifications, additions, or formatting changes.

Examples:
- If tool returns: "SEO_BRIEF_GENERATED|Here's your brief...|message"
- You return: "SEO_BRIEF_GENERATED|Here's your brief...|message" (EXACTLY)

For all other responses, be conversational and helpful.

Key guidelines:
- Always use the tool for ANY SEO-related requests
- NEVER modify SEO_BRIEF_GENERATED responses
- Trust the tool to handle information collection and generation
- Focus on being helpful and knowledgeable for non-SEO conversations

Always prioritize using the tool over giving generic SEO advice."""
            }
        )

    def extract_seo_info_from_conversation(self, messages):
        """Extract SEO information from conversation history using Gemini"""
        
        conversation_text = ""
        for msg in messages:
            role = "User" if msg['sender'] == 'user' else "Assistant"
            conversation_text += f"{role}: {msg['text']}\n"
        
        if not conversation_text.strip():
            print("[DEBUG] No conversation history to extract from")
            return
        
        print(f"[DEBUG] Extracting SEO info from conversation: {conversation_text[:500]}...")
        
        extraction_prompt = f"""
        Analyze this conversation and extract SEO-related information.
        
        Conversation:
        {conversation_text}

        Extract the most recent SEO requirements:
        - content_type: Blog Article, Landing Page, Service Page, Product Description, etc.
        - topic: What the content is about
        - audience: Freelancers, Developers, Designers, Marketers, Business Owners, etc.
        - goal: Increase Traffic, Improve Ranking, Generate Leads, Brand Awareness, etc.
        - tone: Professional, Friendly, Bold, Conversational, Persuasive, etc.
        - primary_keyword: Main keyword or phrase for SEO targeting
        - target_location: City/region for local SEO (if mentioned)
        - competitor_sites: Competitor websites mentioned (if any)

        Return ONLY valid JSON:
        {{"content_type": "extracted or null", "topic": "topic or null", "audience": "audience or null", "goal": "goal or null", "tone": "tone or null", "primary_keyword": "keyword or null", "target_location": "location or null", "competitor_sites": "competitors or null"}}
        """
        
        try:
            response = self.llm.invoke(extraction_prompt)
            
            extracted_text = response.content.strip()
            print(f"[DEBUG] SEO extraction result: {extracted_text}")
            
            # Clean up markdown
            if extracted_text.startswith('```json'):
                extracted_text = extracted_text.replace('```json', '').replace('```', '').strip()
            elif extracted_text.startswith('```'):
                extracted_text = extracted_text.replace('```', '').strip()
            
            extracted_info = json.loads(extracted_text)
            
            # Update seo_info with extracted information
            for key, value in extracted_info.items():
                if key in self.seo_info and value and value.lower() not in ["null", "", "none"]:
                    self.seo_info[key] = value
                    print(f"[DEBUG] Updated {key}: {value}")
            
            print(f"[DEBUG] Final seo_info: {self.seo_info}")
            
        except Exception as e:
            print(f"[DEBUG] SEO extraction error: {e}")
            
            # Manual fallback
            print("[DEBUG] Using manual SEO info search")
            for msg in reversed(messages):
                text = msg['text'].lower()
                
                # Look for content types
                content_types = {
                    "blog": "Blog Article",
                    "landing page": "Landing Page", 
                    "service page": "Service Page",
                    "product page": "Product Description",
                    "case study": "Case Study",
                    "homepage": "Homepage"
                }
                
                for keyword, content_type in content_types.items():
                    if keyword in text:
                        self.seo_info['content_type'] = content_type
                        print(f"[DEBUG] Manual extraction - content_type: {content_type}")
                        break

    def extract_from_current_input(self, user_input: str):
        """Extract SEO information from current user input using Gemini"""
        
        current_info_text = ""
        if any(v for v in self.seo_info.values()):
            current_info_text = f"Current SEO project:\n"
            for key, value in self.seo_info.items():
                if value:
                    current_info_text += f"- {key}: {value}\n"
            current_info_text += "\n"
        
        extraction_prompt = f"""
        {current_info_text}User input: "{user_input}"
        
        Extract SEO information and return ONLY valid JSON:
        
        Content type mapping:
        - "blog article/post" → "Blog Article"
        - "landing page" → "Landing Page"
        - "service page" → "Service Page"
        - "product page/description" → "Product Description"
        - "case study" → "Case Study"
        - "homepage" → "Homepage"
        
        Goal mapping:
        - "traffic" → "Increase Traffic"
        - "ranking" → "Improve Ranking"
        - "leads" → "Generate Leads"
        - "awareness" → "Brand Awareness"
        - "conversions" → "Increase Conversions"
        
        Return ONLY JSON - no explanations:
        {{"content_type": "mapped type or null", "topic": "topic or null", "audience": "audience or null", "goal": "mapped goal or null", "tone": "tone or null", "primary_keyword": "keyword or null", "target_location": "location or null"}}
        """
        
        try:
            response = self.llm.invoke(extraction_prompt)
            
            extracted_text = response.content.strip()
            print(f"[DEBUG] Current input SEO extraction: {extracted_text}")
            
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
            
            # Update seo_info with new information
            for key, value in extracted_info.items():
                if key in self.seo_info and value and value.lower() not in ["null", "", "none"]:
                    self.seo_info[key] = value
                    print(f"[DEBUG] Updated from current input - {key}: {value}")
            
            print(f"[DEBUG] Final seo_info after current input: {self.seo_info}")
                    
        except Exception as e:
            print(f"[DEBUG] Current input SEO extraction error: {e}")
            
            # Simple fallback
            print("[DEBUG] Using simple keyword fallback")
            user_lower = user_input.lower()
            
            # Detect content types
            if "blog" in user_lower:
                self.seo_info['content_type'] = "Blog Article"
            elif "landing page" in user_lower:
                self.seo_info['content_type'] = "Landing Page"
            elif "service" in user_lower:
                self.seo_info['content_type'] = "Service Page"
            elif "product" in user_lower:
                self.seo_info['content_type'] = "Product Description"
            
            print(f"[DEBUG] Fallback extracted content type: {self.seo_info.get('content_type')}")

    def load_seo_preferences(self):
        """Load SEO preferences from User.seoPreferences field"""
        if not self.user_id:
            return
            
        try:
            seo_prefs = MongoDB.get_user_seo_preferences(self.user_id)
            
            # Update seo_info with saved preferences
            for key, value in seo_prefs.items():
                if key in self.seo_info and value:
                    self.seo_info[key] = value
                    print(f"[DEBUG] Loaded SEO preference {key}: {value}")
            
            print(f"[DEBUG] Loaded SEO preferences: {self.seo_info}")
                
        except Exception as e:
            print(f"[DEBUG] Error loading SEO preferences: {e}")

    def save_seo_preferences(self):
        """Save current seo_info to User.seoPreferences field"""
        if not self.user_id:
            return
            
        try:
            # Filter out None values and add timestamp
            seo_prefs_data = {k: v for k, v in self.seo_info.items() if v is not None}
            seo_prefs_data["lastUpdated"] = datetime.utcnow().isoformat()
            
            success = MongoDB.update_user_seo_preferences(self.user_id, seo_prefs_data)
            
            if success:
                print(f"[DEBUG] Saved SEO preferences: {seo_prefs_data}")
            else:
                print("[DEBUG] Failed to save SEO preferences")
                
        except Exception as e:
            print(f"[DEBUG] Error saving SEO preferences: {e}")

    def detect_seo_intent(self, user_input: str) -> bool:
        """Detect if user wants SEO help using Gemini"""
        
        intent_prompt = f"""
        Analyze this user message and determine if they want SEO help or content brief NOW.
        
        User message: "{user_input}"
        
        Return ONLY "YES" if they want SEO help now, or "NO" if they're just providing information.
        
        Examples that mean YES (wants SEO help):
        - "create SEO brief" → YES
        - "optimize for keyword" → YES  
        - "improve my ranking" → YES
        - "SEO strategy for blog" → YES
        - "meta description for..." → YES
        - "keyword research" → YES
        - "SEO audit" → YES
        
        Examples that mean NO (just providing info):
        - "my website is about tech" → NO
        - "I target developers" → NO
        - "my business is in NYC" → NO
        
        Answer: """
        
        try:
            response = self.llm.invoke(intent_prompt)
            
            intent = response.content.strip().upper()
            print(f"[DEBUG] SEO intent detected: {intent} for input: '{user_input}'")
            
            return intent == "YES"
            
        except Exception as e:
            print(f"[DEBUG] Intent detection error: {e}")
            # Enhanced fallback keywords for SEO
            seo_keywords = [
                "seo", "optimize", "ranking", "keyword", "meta", "search", 
                "traffic", "brief", "audit", "strategy", "organic", "serp"
            ]
            
            return any(phrase in user_input.lower() for phrase in seo_keywords)

    def smart_seo_generator(self, user_input: str = "") -> str:
        """Smart SEO brief generation with automatic information collection"""
        
        print(f"[DEBUG] smart_seo_generator called with input: '{user_input}'")
        print(f"[DEBUG] Initial seo_info: {self.seo_info}")
        
        self.user_context = user_input
        
        # Process input
        processed_input = user_input
        if user_input.startswith('{"') and user_input.endswith('}'):
            try:
                parsed_input = json.loads(user_input)
                processed_input = parsed_input.get("description", "create SEO brief")
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
            self.extract_seo_info_from_conversation(recent_messages)
        
        # Extract from current input
        if processed_input and processed_input.strip():
            self.extract_from_current_input(processed_input)
        
        # Save updated seo_info
        self.save_seo_preferences()
        
        print(f"[DEBUG] Current seo_info after extraction: {self.seo_info}")
        
        # Check missing information
        missing_info = [k for k, v in self.seo_info.items() if not v]
        provided_info = {k: v for k, v in self.seo_info.items() if v}
        
        print(f"[DEBUG] Missing info: {missing_info}")
        print(f"[DEBUG] Provided info: {provided_info}")
        
        # Check if we have minimum info to generate SEO brief
        has_minimum_info = (
            self.seo_info.get("content_type") and 
            self.seo_info.get("topic") and
            self.seo_info.get("primary_keyword")
        )
        
        # Detect SEO intent
        wants_seo_help = self.detect_seo_intent(user_input) if user_input else False
        
        # Generate SEO brief if requested and we have minimum info
        if wants_seo_help and has_minimum_info:
            # Auto-complete missing fields
            if missing_info:
                print("[DEBUG] Auto-completing missing fields for SEO brief generation...")
                auto_completed = self.intelligent_auto_complete(self.seo_info.copy())
                for key, value in auto_completed.items():
                    if not self.seo_info.get(key):
                        self.seo_info[key] = value
                self.save_seo_preferences()
            
            print(f"[DEBUG] Generating SEO brief...")
            seo_result = self.generate_seo_brief_with_gemini(self.seo_info, user_context=user_input)
            
            if seo_result["type"] == "seo_brief_generated":
                self.last_generated_brief = seo_result["brief"]
                return f"""{seo_result['brief']}\n{seo_result['message']}"""
            else:
                return seo_result["message"]
        
        # Ask for missing information if not enough to generate
        if not self.seo_info.get("content_type"):
            return self.ask_for_content_type()
        elif not self.seo_info.get("topic"):
            return self.ask_for_topic()
        elif not self.seo_info.get("primary_keyword"):
            return self.ask_for_keyword()
        else:
            # Use natural conversation collection
            return self.collect_seo_info(missing_info, provided_info)

    def intelligent_auto_complete(self, provided_info: dict):
        """Auto-complete missing SEO information using Gemini"""
        
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
        Based on the following SEO information, intelligently suggest appropriate values for the missing fields.
        
        Known information:
        {known_info}
        
        Please complete this JSON with intelligent defaults for the missing fields: {missing_fields}
        
        Guidelines:
        - For audience: Match the content type and topic
        - For goal: Align with content type purpose (blog=traffic, landing=conversions)
        - For tone: Consider the audience and content type
        - For target_location: Only if local business context is clear
        
        Return ONLY a complete JSON object with all fields:
        {{"content_type": "value", "topic": "value", "audience": "value", "goal": "value", "tone": "value", "primary_keyword": "value", "target_location": "value or null", "competitor_sites": "value or null"}}
        """
        
        try:
            response = self.llm.invoke(completion_prompt)
            
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

    def generate_seo_brief_with_gemini(self, info: dict, user_context: str = ""):
        """Generate comprehensive SEO brief using Gemini"""
        
        content_type = info.get('content_type', 'Blog Article')
        topic = info.get('topic', 'General topic')
        audience = info.get('audience', 'General audience')
        goal = info.get('goal', 'Improve Ranking')
        tone = info.get('tone', 'Professional')
        primary_keyword = info.get('primary_keyword', '')
        target_location = info.get('target_location', '')
        competitor_sites = info.get('competitor_sites', '')
        
        # Build comprehensive SEO brief prompt
        brief_prompt = f"""Create a comprehensive SEO content brief for the following:

Content Type: {content_type}
Topic: {topic}
Primary Keyword: {primary_keyword}
Target Audience: {audience}
SEO Goal: {goal}
Tone: {tone}
{f"Target Location: {target_location}" if target_location else ""}
{f"Competitor Sites: {competitor_sites}" if competitor_sites else ""}
{f"Additional Context: {user_context}" if user_context else ""}

Create a detailed SEO brief in Markdown format that includes:

1. **Primary Keyword & Secondary Keywords**
2. **Optimized Meta Title** (under 60 characters)
3. **Meta Description** (under 160 characters)
4. **Suggested Heading Structure** (H1, H2, H3)
5. **Content Outline** with key points to cover
6. **Internal Linking Opportunities**
7. **External Authority Links** to include
8. **SEO Optimization Tips** specific to this content
9. **Target Word Count** and content length recommendations
10. **Featured Snippet Opportunities**

Make it actionable and specific to the topic and keyword provided."""
        
        try:
            print(f"[DEBUG] Generating SEO brief with Gemini...")
            print(f"[DEBUG] Content Type: {content_type}")
            print(f"[DEBUG] Topic: {topic}")
            print(f"[DEBUG] Primary Keyword: {primary_keyword}")
            print(f"[DEBUG] Goal: {goal}")
            
            response = self.llm.invoke(brief_prompt)
            
            generated_brief = response.content.strip()
            print(f"[DEBUG] Successfully generated SEO brief!")
            
            # Create success message
            success_message = f"🎯 **Your SEO Content Brief is ready!**\n\nI've created a comprehensive SEO brief for your {content_type} about {topic}. The brief is optimized for the keyword '{primary_keyword}' and tailored for {audience}."
            
            # Add goal-specific tips
            goal_tips = {
                "Increase Traffic": "\n\n💡 **Traffic Tip**: Focus on long-tail variations and question-based keywords for better organic reach!",
                "Improve Ranking": "\n\n💡 **Ranking Tip**: Optimize for user intent and include semantic keywords to boost SERP position!",
                "Generate Leads": "\n\n💡 **Conversion Tip**: Include clear CTAs and optimize for commercial intent keywords!",
                "Brand Awareness": "\n\n💡 **Awareness Tip**: Target branded and industry-specific terms to increase visibility!"
            }
            
            if goal in goal_tips:
                success_message += goal_tips[goal]
            
            return {
                "type": "seo_brief_generated",
                "message": success_message,
                "brief": generated_brief,
                "content_type": content_type,
                "seo_info": info
            }
            
        except Exception as e:
            print(f"[DEBUG] Gemini SEO brief generation error: {str(e)}")
            return {
                "type": "error",
                "message": f"I encountered an issue generating your SEO brief: {str(e)}. Let me try again with different parameters.",
                "brief": None,
                "seo_info": info
            }

    def ask_for_content_type(self) -> str:
        """Ask user for content type"""
        return """🎯 **What type of content do you want to optimize for SEO?**

Choose from these options:

📝 **Content Types:**
• Blog Article
• Landing Page
• Service Page
• Product Description
• Case Study
• Homepage

💡 **Just let me know which type you're working with, and I'll help you create a comprehensive SEO brief!**"""

    def ask_for_topic(self) -> str:
        """Ask user for topic"""
        content_type = self.seo_info.get('content_type', 'content')
        return f"""Great choice! Now, what's the main topic for your **{content_type}**?

📋 **Topic Examples:**
• Digital marketing strategies
• Web development services
• E-commerce solutions
• Content creation tools
• SEO best practices

🎯 **Just describe what your {content_type.lower()} will be about, and I'll help optimize it for search engines!**"""

    def ask_for_keyword(self) -> str:
        """Ask user for primary keyword"""
        content_type = self.seo_info.get('content_type', 'content')
        topic = self.seo_info.get('topic', 'topic')
        return f"""Perfect! Now, what's your **primary keyword** for this {content_type} about {topic}?

🔍 **Keyword Examples:**
• "best digital marketing tools"
• "web development services NYC"
• "how to improve SEO ranking"
• "e-commerce platform comparison"

💡 **Tip**: Choose a keyword that your target audience would actually search for. It can be 1-4 words long!"""

    def collect_seo_info(self, missing_info: list, provided_info: dict) -> str:
        """Generate natural conversation responses for collecting missing SEO info"""
        
        # If we have minimum info, we can generate with smart defaults
        if provided_info.get("content_type") and provided_info.get("topic") and provided_info.get("primary_keyword"):
            print("[DEBUG] Have minimum info, auto-completing and generating...")
            auto_completed = self.intelligent_auto_complete(self.seo_info.copy())
            for key, value in auto_completed.items():
                if not self.seo_info.get(key):
                    self.seo_info[key] = value
            self.save_seo_preferences()
            
            # Generate the SEO brief
            seo_result = self.generate_seo_brief_with_gemini(self.seo_info)
            
            if seo_result["type"] == "seo_brief_generated":
                self.last_generated_brief = seo_result["brief"]
                return f"""{seo_result['brief']}\n{seo_result['message']}"""
            else:
                return seo_result["message"]
        
        # Ask for next missing field naturally
        content_type = provided_info.get('content_type', 'content')
        
        if 'audience' in missing_info:
            return f"""Great! Who is your target audience for this **{content_type}**?

👥 **Audience Options:**
• **Freelancers** - Independent professionals
• **Developers** - Software engineers and programmers
• **Designers** - Graphic and web designers
• **Marketers** - Digital marketing professionals
• **Business Owners** - Small to medium business owners
• **General Public** - Broad consumer audience

🎯 **Who are you trying to reach with your SEO strategy?**"""
        
        elif 'goal' in missing_info:
            return f"""Excellent! What's your main SEO goal for this **{content_type}**?

🎯 **SEO Goals:**
• **Increase Traffic** - Drive more organic visitors
• **Improve Ranking** - Rank higher in search results
• **Generate Leads** - Convert visitors to customers
• **Brand Awareness** - Increase brand visibility
• **Increase Conversions** - Boost sales or sign-ups

💡 **What do you want to achieve with this SEO optimization?**"""
        
        elif 'tone' in missing_info:
            return f"""Perfect! What tone should your **{content_type}** have?

✍️ **Tone Options:**
• **Professional** - Formal and authoritative
• **Friendly** - Warm and approachable
• **Bold** - Confident and attention-grabbing
• **Conversational** - Casual and relatable
• **Persuasive** - Compelling and sales-focused

🎨 **What tone best fits your brand and audience?**"""
        
        else:
            # Generate with what we have
            return "I have enough information to create a comprehensive SEO brief for you! Let me generate it now. 🚀"

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
        """Handle user query using LangChain agent with Gemini"""
        
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
            agent_type="seo-specialist", 
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
            # Use LangChain agent with Gemini
            print(f"[DEBUG] Using LangChain agent with Gemini")
            ai_response = self.agent.run(query)
            
            # Check if LangChain corrupted a SEO_BRIEF_GENERATED response
            if self.last_generated_brief and "SEO_BRIEF_GENERATED|" not in ai_response:
                print(f"[DEBUG] LangChain corrupted the SEO_BRIEF_GENERATED response")
                print(f"[DEBUG] Original response: {ai_response}")
                
                # Reconstruct the proper format
                if self.last_generated_brief in ai_response:
                    success_message = "I've created a comprehensive SEO brief that perfectly matches your requirements."
                    ai_response = f"{self.last_generated_brief}\n{success_message}"
                    print(f"[DEBUG] Reconstructed proper format: {ai_response}")
            
        except Exception as e:
            print(f"LangChain agent error: {e}, falling back to Gemini direct")
            
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

            # Fallback to Gemini direct
            try:
                response = self.llm.invoke(f"{self.system_prompt}\n\n{enhanced_query}")
                
                ai_response = response.content
                print(f"[DEBUG] Using Gemini direct approach (fallback)")
                
            except Exception as e2:
                print(f"Both LangChain and Gemini failed: {e2}")
                ai_response = "I'm experiencing technical difficulties with my SEO analysis tools. Please try again in a moment."

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
            agent_type="seo-specialist", 
            role="agent", 
            text=ai_response,
            user_id=self.user_id,
            conversation_id=self.conversation_id
        )
        
        return ai_response

def get_seo_specialist_agent(user_id: str = None, conversation_id: str = None):
    return SEOSpecialistAgent(user_id, conversation_id)