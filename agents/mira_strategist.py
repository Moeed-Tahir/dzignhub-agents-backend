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

def search_strategy_conversations(query: str, user_id: str, agent_type: str = "strategist", top_k: int = 10):
    """Search strategy conversations"""
    try:
        print(f"[DEBUG] Searching strategy conversations for: '{query}' (user: {user_id})")
        
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
        
        print(f"[DEBUG] Returning {len(search_results)} strategy conversation results")
        return search_results
        
    except Exception as e:
        print(f"[DEBUG] Search error: {e}")
        return []

# ---------------------------
# Mira Strategist Agent
# ---------------------------
class MiraStrategistAgent:
    def __init__(self, user_id: str = None, conversation_id: str = None):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.agent_name = "strategist"
        self.last_generated_brief = None
        
        # Strategy info structure (based on strategist-mira.js)
        self.strategy_info = {
            "vision": None,          # Big picture idea or mission
            "audience": None,        # Who they want to serve
            "goal": None,           # Short-term goal or success milestone
            "challenge": None,       # Biggest blocker or friction
            "positioning": None,     # Unique message or framing
            "success_metric": None   # A measurable outcome
        }
        
        self.system_prompt = """You are Mira, a thoughtful, strategic assistant who helps users define their growth path by clarifying their goals, audience, and core approach. You operate step-by-step, asking one question at a time, and produce well-structured strategic briefs.

Your capabilities:
🎯 **Strategic Planning:**
- Business strategy development
- Goal clarification and roadmapping
- Audience analysis and targeting
- Positioning and messaging strategy

📊 **Strategic Analysis:**
- Challenge identification
- Success metrics definition
- Risk assessment
- Action planning

🚀 **Strategic Briefs:**
- Comprehensive strategy documents
- Clear vision and goal statements
- Actionable recommendations
- Implementation roadmaps

CRITICAL INSTRUCTIONS:
1. When users request strategic help, use the Generate_Strategy_Brief tool to collect information and create comprehensive briefs
2. When the tool returns "STRATEGY_BRIEF_GENERATED|", you MUST return that EXACT response without modifications
3. For non-strategy requests, be conversational and helpful

Key guidelines:
- Extract multiple values from single messages when possible
- Only ask questions for missing fields
- Never ask again for fields already provided
- Create actionable, realistic strategic briefs
- Focus on practical implementation

Always prioritize using the tool for strategy-related requests."""

        # Use Anthropic Claude
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            temperature=0.7,
            api_key=ANTHROPIC_API
        )
        
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Load conversation history and strategy preferences if exists
        if self.conversation_id and self.user_id:
            self.load_conversation_history()
            self.load_strategy_preferences()

        # Create strategy generation tool
        tools = [
            Tool(
                name="Generate_Strategy_Brief",
                func=self.smart_strategy_generator,
                description="""Use this tool for ALL strategy-related requests including:
                
                BUSINESS STRATEGY: Strategic planning, goal setting, business direction
                GROWTH STRATEGY: Scaling plans, market expansion, growth roadmaps
                POSITIONING: Brand positioning, competitive strategy, market positioning
                PLANNING: Strategic planning, roadmapping, milestone planning
                
                This tool handles information collection AND strategy brief generation for ALL strategic requests.
                Always use this tool for any strategy-related requests."""
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
                'system_message': """You are Mira, a professional strategic planning assistant who helps users create comprehensive business strategies.

Your capabilities:
- Strategic planning and goal setting
- Audience analysis and positioning
- Challenge identification and solution planning
- Success metrics and measurement planning

When users mention wanting ANY strategic help, immediately use the Generate_Strategy_Brief tool.

CRITICAL INSTRUCTION: When the tool returns a response that starts with "STRATEGY_BRIEF_GENERATED|", you MUST return that EXACT response without any modifications, additions, or formatting changes.

Examples:
- If tool returns: "STRATEGY_BRIEF_GENERATED|Here's your brief...|message"
- You return: "STRATEGY_BRIEF_GENERATED|Here's your brief...|message" (EXACTLY)

For all other responses, be conversational and strategic.

Key guidelines:
- Always use the tool for ANY strategy-related requests
- NEVER modify STRATEGY_BRIEF_GENERATED responses
- Trust the tool to handle information collection and generation
- Focus on being thoughtful and strategic for non-strategy conversations

Always prioritize using the tool over giving generic advice."""
            }
        )

    def extract_strategy_info_from_conversation(self, messages):
        """Extract strategy information from conversation history using Claude"""
        
        conversation_text = ""
        for msg in messages:
            role = "User" if msg['sender'] == 'user' else "Assistant"
            conversation_text += f"{role}: {msg['text']}\n"
        
        if not conversation_text.strip():
            print("[DEBUG] No conversation history to extract from")
            return
        
        print(f"[DEBUG] Extracting strategy info from conversation: {conversation_text[:500]}...")
        
        extraction_prompt = f"""
        Analyze this conversation and extract strategic planning information.
        
        Conversation:
        {conversation_text}

        Extract the most recent strategic requirements:
        - vision: Big picture idea, mission, or purpose
        - audience: Target audience, customers, or users they want to serve
        - goal: Short-term goal, milestone, or success target
        - challenge: Biggest blocker, friction, or problem they're facing
        - positioning: Unique message, value proposition, or competitive advantage
        - success_metric: How they'll measure success (numbers, metrics, outcomes)

        Return ONLY valid JSON:
        {{"vision": "extracted vision or null", "audience": "target audience or null", "goal": "specific goal or null", "challenge": "main challenge or null", "positioning": "unique positioning or null", "success_metric": "success measurement or null"}}
        """
        
        try:
            response = self.llm.invoke(extraction_prompt)
            
            extracted_text = response.content.strip()
            print(f"[DEBUG] Strategy extraction result: {extracted_text}")
            
            # Clean up markdown
            if extracted_text.startswith('```json'):
                extracted_text = extracted_text.replace('```json', '').replace('```', '').strip()
            elif extracted_text.startswith('```'):
                extracted_text = extracted_text.replace('```', '').strip()
            
            extracted_info = json.loads(extracted_text)
            
            # Update strategy_info with extracted information
            for key, value in extracted_info.items():
                if key in self.strategy_info and value and value.lower() not in ["null", "", "none"]:
                    self.strategy_info[key] = value
                    print(f"[DEBUG] Updated {key}: {value}")
            
            print(f"[DEBUG] Final strategy_info: {self.strategy_info}")
            
        except Exception as e:
            print(f"[DEBUG] Strategy extraction error: {e}")
            
            # Manual fallback
            print("[DEBUG] Using manual strategy info search")
            for msg in reversed(messages):
                text = msg['text'].lower()
                
                # Look for strategy keywords
                strategy_keywords = {
                    "vision": ["vision", "mission", "purpose", "big picture"],
                    "goal": ["goal", "target", "milestone", "achieve", "want to"],
                    "challenge": ["challenge", "problem", "blocker", "struggle", "difficult"],
                    "audience": ["audience", "customers", "users", "serve", "target"],
                    "positioning": ["positioning", "unique", "different", "advantage", "value prop"],
                    "success_metric": ["metric", "measure", "kpi", "success", "track", "number"]
                }
                
                for field, keywords in strategy_keywords.items():
                    for keyword in keywords:
                        if keyword in text and not self.strategy_info[field]:
                            # Extract sentence containing the keyword
                            sentences = msg['text'].split('.')
                            for sentence in sentences:
                                if keyword in sentence.lower():
                                    self.strategy_info[field] = sentence.strip()
                                    print(f"[DEBUG] Manual extraction - {field}: {sentence.strip()}")
                                    break
                            break

    def extract_from_current_input(self, user_input: str):
        """Extract strategy information from current user input using Claude"""
        
        current_info_text = ""
        if any(v for v in self.strategy_info.values()):
            current_info_text = f"Current strategy project:\n"
            for key, value in self.strategy_info.items():
                if value:
                    current_info_text += f"- {key}: {value}\n"
            current_info_text += "\n"
        
        extraction_prompt = f"""
        {current_info_text}User input: "{user_input}"
        
        Extract strategic planning information and return ONLY valid JSON:
        
        Field definitions:
        - vision: Big picture purpose, mission, or what they want to build/achieve
        - audience: Target customers, users, or people they want to serve
        - goal: Specific short-term objective, milestone, or measurable target
        - challenge: Main problem, blocker, or difficulty they're facing
        - positioning: How they're different, unique value, or competitive advantage
        - success_metric: How they'll measure success (numbers, KPIs, metrics)
        
        Return ONLY JSON - no explanations:
        {{"vision": "extracted or null", "audience": "extracted or null", "goal": "extracted or null", "challenge": "extracted or null", "positioning": "extracted or null", "success_metric": "extracted or null"}}
        """
        
        try:
            response = self.llm.invoke(extraction_prompt)
            
            extracted_text = response.content.strip()
            print(f"[DEBUG] Current input strategy extraction: {extracted_text}")
            
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
            
            # Update strategy_info with new information
            for key, value in extracted_info.items():
                if key in self.strategy_info and value and value.lower() not in ["null", "", "none"]:
                    self.strategy_info[key] = value
                    print(f"[DEBUG] Updated from current input - {key}: {value}")
            
            print(f"[DEBUG] Final strategy_info after current input: {self.strategy_info}")
                    
        except Exception as e:
            print(f"[DEBUG] Current input strategy extraction error: {e}")
            
            # Simple fallback
            print("[DEBUG] Using simple keyword fallback")
            user_lower = user_input.lower()
            
            # Detect strategy elements
            if any(word in user_lower for word in ["goal", "want to", "achieve", "target"]):
                if not self.strategy_info['goal']:
                    self.strategy_info['goal'] = user_input
            elif any(word in user_lower for word in ["vision", "mission", "purpose", "build"]):
                if not self.strategy_info['vision']:
                    self.strategy_info['vision'] = user_input
            elif any(word in user_lower for word in ["challenge", "problem", "struggle", "difficult"]):
                if not self.strategy_info['challenge']:
                    self.strategy_info['challenge'] = user_input
            
            print(f"[DEBUG] Fallback extracted info: {self.strategy_info}")

    def load_strategy_preferences(self):
        """Load strategy preferences from User.strategyPreferences field"""
        if not self.user_id:
            return
            
        try:
            strategy_prefs = MongoDB.get_user_strategy_preferences(self.user_id)
            
            # Update strategy_info with saved preferences
            for key, value in strategy_prefs.items():
                if key in self.strategy_info and value:
                    self.strategy_info[key] = value
                    print(f"[DEBUG] Loaded strategy preference {key}: {value}")
            
            print(f"[DEBUG] Loaded strategy preferences: {self.strategy_info}")
                
        except Exception as e:
            print(f"[DEBUG] Error loading strategy preferences: {e}")

    def save_strategy_preferences(self):
        """Save current strategy_info to User.strategyPreferences field"""
        if not self.user_id:
            return
            
        try:
            # Filter out None values and add timestamp
            strategy_prefs_data = {k: v for k, v in self.strategy_info.items() if v is not None}
            strategy_prefs_data["lastUpdated"] = datetime.utcnow().isoformat()
            
            success = MongoDB.update_user_strategy_preferences(self.user_id, strategy_prefs_data)
            
            if success:
                print(f"[DEBUG] Saved strategy preferences: {strategy_prefs_data}")
            else:
                print("[DEBUG] Failed to save strategy preferences")
                
        except Exception as e:
            print(f"[DEBUG] Error saving strategy preferences: {e}")

    def detect_strategy_intent(self, user_input: str) -> bool:
        """Detect if user wants strategic help using Claude"""
        
        intent_prompt = f"""
        Analyze this user message and determine if they want strategic planning help NOW.
        
        User message: "{user_input}"
        
        Return ONLY "YES" if they want strategic help now, or "NO" if they're just providing information.
        
        Examples that mean YES (wants strategy help):
        - "create strategy plan" → YES
        - "help me plan my business" → YES  
        - "need strategic guidance" → YES
        - "build a roadmap" → YES
        - "strategic planning" → YES
        - "business strategy" → YES
        - "growth strategy" → YES
        
        Examples that mean NO (just providing info):
        - "my business is about tech" → NO
        - "I target developers" → NO
        - "my goal is growth" → NO
        
        Answer: """
        
        try:
            response = self.llm.invoke(intent_prompt)
            
            intent = response.content.strip().upper()
            print(f"[DEBUG] Strategy intent detected: {intent} for input: '{user_input}'")
            
            return intent == "YES"
            
        except Exception as e:
            print(f"[DEBUG] Intent detection error: {e}")
            # Enhanced fallback keywords for strategy
            strategy_keywords = [
                "strategy", "strategic", "plan", "planning", "roadmap", 
                "goal", "vision", "mission", "growth", "business plan"
            ]
            
            return any(phrase in user_input.lower() for phrase in strategy_keywords)

    def smart_strategy_generator(self, user_input: str = "") -> str:
        """Smart strategy brief generation with automatic information collection"""
        
        print(f"[DEBUG] smart_strategy_generator called with input: '{user_input}'")
        print(f"[DEBUG] Initial strategy_info: {self.strategy_info}")
        
        self.user_context = user_input
        
        # Process input
        processed_input = user_input
        if user_input.startswith('{"') and user_input.endswith('}'):
            try:
                parsed_input = json.loads(user_input)
                processed_input = parsed_input.get("description", "create strategy")
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
            self.extract_strategy_info_from_conversation(recent_messages)
        
        # Extract from current input
        if processed_input and processed_input.strip():
            self.extract_from_current_input(processed_input)
        
        # Save updated strategy_info
        self.save_strategy_preferences()
        
        print(f"[DEBUG] Current strategy_info after extraction: {self.strategy_info}")
        
        # Check missing information
        missing_info = [k for k, v in self.strategy_info.items() if not v]
        provided_info = {k: v for k, v in self.strategy_info.items() if v}
        
        print(f"[DEBUG] Missing info: {missing_info}")
        print(f"[DEBUG] Provided info: {provided_info}")
        
        # Check if we have minimum info to generate strategy brief
        has_minimum_info = (
            self.strategy_info.get("vision") and 
            self.strategy_info.get("goal") and
            self.strategy_info.get("audience")
        )
        
        # Detect strategy intent
        wants_strategy_help = self.detect_strategy_intent(user_input) if user_input else False
        
        # Generate strategy brief if requested and we have minimum info
        if wants_strategy_help and has_minimum_info:
            # Auto-complete missing fields
            if missing_info:
                print("[DEBUG] Auto-completing missing fields for strategy brief generation...")
                auto_completed = self.intelligent_auto_complete(self.strategy_info.copy())
                for key, value in auto_completed.items():
                    if not self.strategy_info.get(key):
                        self.strategy_info[key] = value
                self.save_strategy_preferences()
            
            print(f"[DEBUG] Generating strategy brief...")
            strategy_result = self.generate_strategy_brief_with_claude(self.strategy_info, user_context=user_input)
            
            if strategy_result["type"] == "strategy_brief_generated":
                self.last_generated_brief = strategy_result["brief"]
                return f"""STRATEGY_BRIEF_GENERATED|{strategy_result['brief']}|{strategy_result['message']}"""
            else:
                return strategy_result["message"]
        
        # Ask for missing information if not enough to generate
        if not self.strategy_info.get("vision"):
            return self.ask_for_vision()
        elif not self.strategy_info.get("audience"):
            return self.ask_for_audience()
        elif not self.strategy_info.get("goal"):
            return self.ask_for_goal()
        else:
            # Use natural conversation collection
            return self.collect_strategy_info(missing_info, provided_info)

    def intelligent_auto_complete(self, provided_info: dict):
        """Auto-complete missing strategy information using Claude"""
        
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
        Based on the following strategic information, intelligently suggest appropriate values for the missing fields.
        
        Known information:
        {known_info}
        
        Please complete this JSON with intelligent defaults for the missing fields: {missing_fields}
        
        Guidelines:
        - For challenge: Consider common business/startup challenges
        - For positioning: Think about competitive advantages
        - For success_metric: Suggest measurable, relevant KPIs
        - Keep suggestions realistic and actionable
        
        Return ONLY a complete JSON object with all fields:
        {{"vision": "value", "audience": "value", "goal": "value", "challenge": "value", "positioning": "value", "success_metric": "value"}}
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

    def generate_strategy_brief_with_claude(self, info: dict, user_context: str = ""):
        """Generate comprehensive strategy brief using Claude"""
        
        vision = info.get('vision', 'Business vision')
        audience = info.get('audience', 'Target audience')
        goal = info.get('goal', 'Business goal')
        challenge = info.get('challenge', 'Business challenge')
        positioning = info.get('positioning', 'Unique positioning')
        success_metric = info.get('success_metric', 'Success measurement')
        
        # Build comprehensive strategy brief prompt
        brief_prompt = f"""Create a comprehensive strategic brief for the following business/project:

Vision: {vision}
Target Audience: {audience}
Primary Goal: {goal}
Biggest Challenge: {challenge}
Positioning: {positioning}
Success Metric: {success_metric}
{f"Additional Context: {user_context}" if user_context else ""}

Create a detailed strategy brief in Markdown format that includes:

## Strategy Brief: [Vision Summary]

**Vision**: {vision}

**Primary Goal**: {goal}

**Target Audience**: {audience}

**Biggest Challenge**: {challenge}

**Positioning Statement**: {positioning}

**Success Metric**: {success_metric}

**Key Actions**:
- [Provide 3-4 specific, actionable steps]

**Risks & Watchouts**:
- [Identify 2-3 potential risks or challenges]

**Strategic Recommendation**: 
[Provide focused, actionable advice for moving forward]

**Next Steps**:
- [List immediate next steps to take]

Make it actionable, realistic, and tailored to their specific situation."""
        
        try:
            print(f"[DEBUG] Generating strategy brief with Claude...")
            print(f"[DEBUG] Vision: {vision}")
            print(f"[DEBUG] Goal: {goal}")
            print(f"[DEBUG] Audience: {audience}")
            print(f"[DEBUG] Challenge: {challenge}")
            
            response = self.llm.invoke(brief_prompt)
            
            generated_brief = response.content.strip()
            print(f"[DEBUG] Successfully generated strategy brief!")
            
            # Create success message
            success_message = f"🎯 **Your Strategic Brief is ready!**\n\nI've created a comprehensive strategy plan for your {vision}. The brief includes clear actions, risk analysis, and recommendations to help you achieve {goal}."
            
            # Add goal-specific tips
            goal_tips = {
                "growth": "\n\n💡 **Growth Tip**: Focus on sustainable scaling strategies and track leading indicators!",
                "launch": "\n\n💡 **Launch Tip**: Build momentum with early adopters before broader market entry!",
                "revenue": "\n\n💡 **Revenue Tip**: Prioritize high-impact, quick-win revenue streams first!",
                "market": "\n\n💡 **Market Tip**: Validate assumptions with real customer feedback before major investments!"
            }
            
            goal_lower = goal.lower()
            for keyword, tip in goal_tips.items():
                if keyword in goal_lower:
                    success_message += tip
                    break
            
            return {
                "type": "strategy_brief_generated",
                "message": success_message,
                "brief": generated_brief,
                "vision": vision,
                "strategy_info": info
            }
            
        except Exception as e:
            print(f"[DEBUG] Claude strategy brief generation error: {str(e)}")
            return {
                "type": "error",
                "message": f"I encountered an issue generating your strategy brief: {str(e)}. Let me try again with different parameters.",
                "brief": None,
                "strategy_info": info
            }

    def ask_for_vision(self) -> str:
        """Ask user for vision"""
        return """🌟 **What's your vision or big picture purpose?**

This is about your mission and what you want to build or achieve:

📋 **Vision Examples:**
• "Help students build better study habits"
• "Make design accessible to small businesses"
• "Create the best project management tool for developers"
• "Build a community for freelance writers"

💡 **Think big picture**: What impact do you want to make? What problem are you solving?"""

    def ask_for_audience(self) -> str:
        """Ask user for target audience"""
        return """👥 **Who are you trying to serve?**

Choose your primary target audience:

**👥 Audience Options:**
• **Students** - University/college students
• **Creators** - Content creators, artists, influencers
• **Startup Founders** - Early-stage entrepreneurs
• **Small Business Owners** - Local business owners
• **Corporate Teams** - Enterprise employees
• **Developers** - Software engineers and programmers
• **General Public** - Broad consumer audience

🎯 **Who is your ideal customer or user?**"""

    def ask_for_goal(self) -> str:
        """Ask user for goal"""
        vision = self.strategy_info.get('vision', 'vision')
        audience = self.strategy_info.get('audience', 'audience')
        return f"""🎯 **What's your most important short-term goal?**

For your {vision} targeting {audience}:

**🎯 Goal Examples:**
• Get 1,000 users in 3 months
• Generate $10k monthly revenue
• Launch MVP by end of quarter
• Build email list of 500 subscribers
• Secure 5 enterprise clients

💡 **Make it specific and measurable**: What do you want to achieve in the next 3-6 months?"""

    def collect_strategy_info(self, missing_info: list, provided_info: dict) -> str:
        """Generate natural conversation responses for collecting missing strategy info"""
        
        # If we have minimum info, we can generate with smart defaults
        if provided_info.get("vision") and provided_info.get("goal") and provided_info.get("audience"):
            print("[DEBUG] Have minimum info, auto-completing and generating...")
            auto_completed = self.intelligent_auto_complete(self.strategy_info.copy())
            for key, value in auto_completed.items():
                if not self.strategy_info.get(key):
                    self.strategy_info[key] = value
            self.save_strategy_preferences()
            
            # Generate the strategy brief
            strategy_result = self.generate_strategy_brief_with_claude(self.strategy_info)
            
            if strategy_result["type"] == "strategy_brief_generated":
                self.last_generated_brief = strategy_result["brief"]
                return f"""STRATEGY_BRIEF_GENERATED|{strategy_result['brief']}|{strategy_result['message']}"""
            else:
                return strategy_result["message"]
        
        # Ask for next missing field naturally
        vision = provided_info.get('vision', 'your project')
        
        if 'challenge' in missing_info:
            return f"""Great! What's your **biggest challenge or roadblock** right now?

🧱 **Common Challenges:**
• **Low retention** - Users sign up but don't stick around
• **Finding customers** - Hard to reach target audience
• **Limited resources** - Time, money, or team constraints
• **Technical challenges** - Building the right solution
• **Market competition** - Standing out from competitors
• **Scaling issues** - Growing without breaking

💭 **What's the main thing holding you back from achieving your goal?**"""
        
        elif 'positioning' in missing_info:
            return f"""Perfect! How would you describe your **positioning or what makes you different**?

💎 **Positioning Examples:**
• "The only design tool built specifically for non-designers"
• "Project management with built-in mental health features"
• "The fastest way to learn coding through real projects"
• "Community-driven learning platform"

🌟 **What's your unique angle or competitive advantage?**"""
        
        elif 'success_metric' in missing_info:
            return f"""Excellent! How will you **measure success**?

📈 **Success Metrics Examples:**
• **User growth**: Monthly active users, sign-ups
• **Revenue**: Monthly recurring revenue, sales
• **Engagement**: Daily usage, retention rate
• **Market**: Market share, brand awareness
• **Operational**: Customer satisfaction, efficiency

📊 **What number or metric will tell you you're winning?**"""
        
        else:
            # Generate with what we have
            return "I have enough strategic information to create a comprehensive plan for you! Let me generate your strategy brief now. 🚀"

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
            agent_type="strategist", 
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
            
            # Check if LangChain corrupted a STRATEGY_BRIEF_GENERATED response
            if self.last_generated_brief and "STRATEGY_BRIEF_GENERATED|" not in ai_response:
                print(f"[DEBUG] LangChain corrupted the STRATEGY_BRIEF_GENERATED response")
                print(f"[DEBUG] Original response: {ai_response}")
                
                # Reconstruct the proper format
                if self.last_generated_brief in ai_response:
                    success_message = "I've created a comprehensive strategic plan that perfectly matches your requirements."
                    ai_response = f"STRATEGY_BRIEF_GENERATED|{self.last_generated_brief}|{success_message}"
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
                ai_response = "I'm experiencing technical difficulties with my strategic analysis tools. Please try again in a moment."

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
            agent_type="strategist", 
            role="agent", 
            text=ai_response,
            user_id=self.user_id,
            conversation_id=self.conversation_id
        )
        
        return ai_response

def get_mira_strategist_agent(user_id: str = None, conversation_id: str = None):
    return MiraStrategistAgent(user_id, conversation_id)