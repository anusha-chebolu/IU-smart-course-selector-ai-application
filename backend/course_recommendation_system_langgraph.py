from typing import Dict, TypedDict, List, Optional, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain.memory import ConversationBufferMemory
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
import json
import os
from dotenv import load_dotenv
import random
import string
from datetime import datetime
from util import get_course_recommendations
import re

# Load environment variables
load_dotenv()

def clean_and_parse_json(llm_output: str):
    # Remove triple backticks and optional language tag (e.g., ```json)
    cleaned = re.sub(r"^```json|```$", "", llm_output.strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(cleaned)
        return parsed
    except json.JSONDecodeError as e:
        print("JSON parsing error:", e)
        return None

# Define output schemas for each agent
class DecisionOutput(BaseModel):
    action: str = Field(description="The action to take: recommendation, inquiry, or clarification_needed")
    career_goal: Optional[List] = Field(default=[], description="The career goal mentioned by the user, if any")
    course_name: Optional[List] = Field(default=[], description="The specific course mentioned by the user, if any")
    course_work: Optional[List] = Field(default=[], description="Course work related to the user's query or career goal, empty list if not applicable")
    original_query: str = Field(description="The original query from the user")
    reasoning: str = Field(description="The reasoning behind the classification")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "action": "recommendation",
                    "career_goal": ["UX Designer"],
                    "course_name": [],
                    "course_work": ["User Interface Design", "Human-Computer Interaction", "UX Research Methods"],
                    "original_query": "I want to become a UX Designer",
                    "reasoning": "The user clearly expresses a career goal using 'I want to become', which aligns with the recommendation action."
                },
                {
                    "action": "recommendation",
                    "career_goal": ["Product Manager"],
                    "course_name": [],
                    "course_work": ["Product Development", "Agile Methodologies", "Market Research"],
                    "original_query": "I'm thinking of switching to Product Management. Any courses I should take?",
                    "reasoning": "The user expresses intent to change their career path to Product Management, which justifies a recommendation."
                },
                {
                    "action": "inquiry",
                    "career_goal": [],
                    "course_name": ["Advanced Database Concepts"],
                    "course_work": ["SQL Optimization", "NoSQL Systems", "Database Design"],
                    "original_query": "What topics are covered in Advanced Database Concepts?",
                    "reasoning": "The user is directly asking about the content of a specific course, which fits the inquiry action."
                },
                {
                    "action": "inquiry",
                    "career_goal": [],
                    "course_name": ["Data Visualization"],
                    "course_work": ["Tableau", "D3.js", "Information Design"],
                    "original_query": "Can you tell me if Data Visualization is offered in the evening?",
                    "reasoning": "The user is inquiring about the timing of a specific course, which is classified as an inquiry."
                },
                {
                    "action": "clarification_needed",
                    "career_goal": [],
                    "course_name": [],
                    "course_work": [],
                    "original_query": "I love beating my friends",
                    "reasoning": "The query is playful and lacks any educational intent, career goal, or course reference. No valid classification can be made."
                },
                {
                    "action": "clarification_needed",
                    "career_goal": [],
                    "course_name": [],
                    "course_work": [],
                    "original_query": "I'm so lost. This semester has been overwhelming.",
                    "reasoning": "The user is expressing emotional distress without providing a clear goal or inquiry. It's unclear what help they need, so clarification is required."
                }
            ]
        }
    }

class CourseRecommendation(BaseModel):
    course_id: str = Field(description="The course id (e.g., 084226)")
    course_code: str = Field(description="The course code (e.g., BUEX-V 594)")
    course_title: str = Field(description="The title of the course")
    course_description: str = Field(description="Brief description of the course")
    skill_development: List[str] = Field(description="Skills that will be developed by taking this course")
    career_alignment: str = Field(description="How this course aligns with the career goal")
    relevance_reasoning: str = Field(
        description="Explanation of why this course is relevant to the user's query or career goal"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "course_id": "096191",
                    "course_code": "CS401",
                    "course_title": "Machine Learning",
                    "course_description": "Introduction to machine learning algorithms and applications",
                    "skill_development": ["Python", "Statistical Analysis", "Algorithm Design"],
                    "career_alignment": "Core skill for data scientists",
                    "relevance_reasoning": "This course directly addresses the user's interest in becoming a data scientist by providing foundational skills in machine learning algorithms, which are essential for processing and analyzing large datasets to extract meaningful insights - a primary responsibility of data scientists."
                },
                {
                    "course_id": "084226",
                    "course_code": "BUEX-V 594",
                    "course_title": "User Experience Design",
                    "course_description": "Principles and methods of designing digital products focused on the user experience",
                    "skill_development": ["Wireframing", "User Testing", "Prototyping"],
                    "career_alignment": "Essential for UX Designer roles",
                    "relevance_reasoning": "This course directly addresses the user's goal of becoming a UX Designer by teaching core user experience methodologies. The hands-on prototyping projects will build a portfolio that employers look for, and the user testing components align perfectly with the user-centered research mentioned in the query."
                },
                {
                    "course_id": "079448",
                    "course_code": "BUS-M 455",
                    "course_title": "Product Management Strategy",
                    "course_description": "Comprehensive overview of product lifecycle management and strategy development",
                    "skill_development": ["Roadmapping", "Stakeholder Management", "Market Analysis"],
                    "career_alignment": "Directly applicable to Product Manager positions",
                    "relevance_reasoning": "Since the user expressed interest in switching to Product Management, this course provides the strategic foundation they'll need. The roadmapping and stakeholder management components address the cross-functional leadership aspects of product management that differentiate it from other roles they might be transitioning from."
                }
            ]
        }
    }

class RecommendationOutput(BaseModel):
    recommended_courses: List[CourseRecommendation] = Field(description="List of 5 recommended courses",min_items=5,
        max_items=5)
    recommendation_strategy: str = Field(description="The overall strategy for the recommendations")
    additional_guidance: Optional[str] = Field(default=None, description="Any additional guidance for the student")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "recommended_courses": [
                        {
                            "course_id": "096191",
                            "course_code": "CS401",
                            "course_title": "Machine Learning",
                            "course_description": "Introduction to machine learning algorithms and applications",
                            "skill_development": ["Python", "Statistical Analysis", "Algorithm Design"],
                            "career_alignment": "Core skill for data scientists",
                            "relevance_reasoning": "This course directly addresses your interest in becoming a data scientist by providing foundational skills in machine learning algorithms, which are essential for processing and analyzing large datasets to extract meaningful insights - a primary responsibility of data scientists."
                        },
                        {
                            "course_id": "084226",
                            "course_code": "BUEX-V 594",
                            "course_title": "User Experience Design",
                            "course_description": "Principles and methods of designing digital products focused on the user experience",
                            "skill_development": ["Wireframing", "User Testing", "Prototyping"],
                            "career_alignment": "Essential for UX Designer roles",
                            "relevance_reasoning": "Based on your interest in human-computer interaction, this course will provide you with practical skills in UX design that complement your technical background. The combination of technical knowledge and user experience skills is highly valued in the industry."
                        }
                    ],
                    "recommendation_strategy": "These recommendations focus on building your technical skills in data science while also expanding your knowledge in user experience design, which aligns with your expressed career interests and complements your computer science background.",
                    "additional_guidance": "Consider taking these courses in sequence, with Machine Learning first to build your technical foundation, followed by User Experience Design to diversify your skill set."
                }
            ]
        }
    }

class GeneralInquiryOutput(BaseModel):
    course_information: dict = Field(description="Information about the requested course")
    additional_details: Optional[str] = Field(default=None, description="Any additional relevant details")
    professor_name: Optional[str] = Field(default=None, description="Name of the professor teaching the course")
    total_strength: Optional[int] = Field(default=None, description="Total number of students that can enroll in the course")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "course_information": {
                        "course_id": "CS301",
                        "course_name": "Advanced Programming",
                        "prerequisites": ["CS101", "CS201"]
                    },
                    "professor_name": "Dr. Jane Smith",
                    "total_strength": 120
                }
            ]
        }
    }

class ClarificationOutput(BaseModel):
    clarification_question: str = Field(description="Question to clarify user's intent")
    possible_intents: List[str] = Field(description="Possible intents the user might have")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "clarification_question": "Could you share what career path you're interested in exploring?",
                    "possible_intents": ["Career guidance", "Course selection", "Degree requirements"]
                },
                {
                    "clarification_question": "I understand you might be feeling overwhelmed. Could you share more about what's making you consider dropping out?",
                    "possible_intents": ["Academic struggles", "Financial concerns", "Personal challenges", "Lack of interest in current program"]
                }
            ]
        }
    }

# Agent prompt templates
DECISION_PROMPT = """
You are a highly intelligent Course Recommendation Assistant.

Your primary task is to analyze user queries and classify them into a structured JSON format with one of the following actions:

1. "recommendation" — If the user mentions a **career goal** (e.g., "I want to become a data scientist"), extract the goal and set the action to "recommendation".
2. "inquiry" — If the user is asking about a **specific course** (e.g., course description, timing, instructor), extract the course name and set the action to "inquiry".
3. "clarification_needed" — If the user's message is vague, lacks a clear goal or course, or is playful/unserious, set the action to "clarification_needed".

Classification Rules:
- If both a career goal and a course name are mentioned, prioritize **"recommendation"**.
- If a new career goal is introduced in a later query, treat it as a **new recommendation**.
- Do **not** assume a career goal unless the user clearly implies it (e.g., "I want to become...", "I'm thinking of becoming...", "I want to switch to...").
- If the user expresses dissatisfaction, frustration, or feeling lost **without a clear next step**, classify the query as **"clarification_needed"**.
- Do **not** infer meaning from jokes, metaphors, or playful language. Only classify based on explicit educational or career-related intent.

User query: {query}

{format_instructions}
"""

RECOMMENDATION_PROMPT = """
You are an Advanced Course Recommendation Agent with deep expertise in academic and career path guidance. Your primary objective is to provide personalized, strategic course recommendations based on a student's:
- Career aspirations
- Academic interests
- Skill development needs
- Holistic educational growth

Core Recommendation Principles:
1. Contextualize Recommendations
- Go beyond surface-level course matching
- Consider long-term career trajectory
- Identify skill gaps and development opportunities
- Create a strategic learning pathway

2. Recommendation Framework
- Provide 5 course recommendations
- Include detailed description for each recommendation
- Connect courses to:
  * Long-term career goals
  * Skill development
  * Interdisciplinary learning

3. Recommendation Structure
For each recommended course, provide:
- course_name and course_title
- course_description
- Skill Development Alignment
- Career Impact

4. Analytical Depth
- Analyze course descriptions
- Consider prerequisite knowledge
- Evaluate potential learning outcomes
- Assess interdisciplinary connections

Career goal: {career_goal}
Available courses: {courses}

{format_instructions}
"""

INQUIRY_PROMPT = """
You are a course information expert at a university. You provide detailed and accurate information about specific courses when students inquire about them.

Please provide comprehensive information about the requested course, including:
- Course description
- Prerequisites (if any)
- Credit hours
- Typical semester offerings
- Related courses that might also interest the student

Course requested: {course_name}
Course details: {course_details}

{format_instructions}
"""

CLARIFICATION_PROMPT = """
You are a helpful academic advisor assistant. The user has provided a query that needs clarification before you can assist them properly.

Please generate a clarifying question that will help determine what the user is looking for. Your response should be friendly, helpful, and guide the user toward providing the information needed to assist them with course selection or academic information.

Original query: {query}

{format_instructions}
"""

SUPERVISOR_PROMPT = """
You are a Course Recommendation System Supervisor responsible for managing the overall conversation with the user and coordinating between specialized agents.

Your responsibilities include:
1. Providing a cohesive user experience
2. Ensuring responses are relevant and helpful
3. Maintaining conversation flow
4. Presenting information in a clear, friendly, and structured manner

Important - 
DO NOT ASK USER COLLEGE DETAILS

Previous conversation context:
{chat_history}

Most recent query: {query}

Agent response to process: {agent_response}

Craft a helpful, natural-sounding response to the user that incorporates the specialized agent's information while maintaining a conversational tone.
"""

TITLE_PROMPT = """
Generate a short, descriptive title (5 words or less) for a chat session about course recommendations that starts with this query:

User Query: {query}

Title:
"""

# Initialize output parsers
decision_parser = PydanticOutputParser(pydantic_object=DecisionOutput)
recommendation_parser = PydanticOutputParser(pydantic_object=RecommendationOutput)
inquiry_parser = PydanticOutputParser(pydantic_object=GeneralInquiryOutput)
clarification_parser = PydanticOutputParser(pydantic_object=ClarificationOutput)

# Initialize LLM
llm = ChatAnthropic(
    model_name=os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
    temperature=0.3,
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=4092,
    timeout=10.0,  # Add timeout
    max_retries=1  # Reduce retries
)

# Define the state for the graph
class AgentState(TypedDict):
    # User and session information
    user_id: str
    session_id: str
    
    # Conversation state
    query: str
    chat_history: List[Dict[str, str]]
    
    # Decision making
    decision: Optional[DecisionOutput]
    action: Optional[str]
    
    # Agent outputs
    raw_agent_response: Optional[str]
    parsed_agent_response: Optional[Dict]
    
    # Final outputs
    final_response: Optional[str]
    chat_title: Optional[str]
    
    # Database session
    db_session: any
    
    # Control flow
    next: str

# Database helper functions (placeholders)
def save_chat_message(user_id, session_id, message_type, content):
    """Save chat message to database"""
    print(f"Saving {message_type} message for user {user_id} in session {session_id}")
    return True

def save_chat_session(session_id, user_id, title, metadata=None):
    """Save chat session metadata to database"""
    session_data = {
        "session_id": session_id,
        "user_id": user_id,
        "title": title,
        "created_at": datetime.now().isoformat(),
        "metadata": metadata or {}
    }
    print(f"Saving session metadata: {json.dumps(session_data, indent=2)}")
    return True

def get_user_sessions(user_id):
    """Get all sessions for a user"""
    return []

# Agent functions for LangGraph
def decision_agent(state: AgentState) -> Dict:
    """Decision agent that determines the intent of the query"""
    decision_prompt = PromptTemplate(
        template=DECISION_PROMPT,
        input_variables=["query"],
        partial_variables={"format_instructions": decision_parser.get_format_instructions()}
    )
    
    result = llm.invoke(decision_prompt.format(query=state["query"]))
    
    try:
        decision = decision_parser.parse(result.content)
        return {
            "decision": decision,
            "action": decision.action,
            "next": decision.action  # This determines the next node to visit
        }
    except Exception as e:
        print(f"Error in decision agent: {e}")
        return {
            "action": "clarification_needed",
            "next": "clarification_needed"
        }

def recommendation_agent(state: AgentState) -> Dict:
    """Recommendation agent for course recommendations"""
    decision = state["decision"]
    courses = get_course_recommendations(
        state["user_id"],
        decision.career_goal,
        state["db_session"]
    )
    
    recommendation_prompt = PromptTemplate(
        template=RECOMMENDATION_PROMPT,
        input_variables=["career_goal", "courses"],
        partial_variables={"format_instructions": recommendation_parser.get_format_instructions()}
    )
    
    result = llm.invoke(
        recommendation_prompt.format(
            career_goal=decision.career_goal,
            courses=json.dumps(courses)
        )
    )
    
    return {
        "raw_agent_response": result.content,
        "next": "supervisor_agent"  # Updated to match new node name
    }

def inquiry_agent(state: AgentState) -> Dict:
    """Inquiry agent for course information"""
    decision = state["decision"]
    course_details = get_course_recommendations(
        state["user_id"],
        decision.course_name,
        state["db_session"]
    )
    
    inquiry_prompt = PromptTemplate(
        template=INQUIRY_PROMPT,
        input_variables=["course_name", "course_details"],
        partial_variables={"format_instructions": inquiry_parser.get_format_instructions()}
    )
    
    result = llm.invoke(
        inquiry_prompt.format(
            course_name=decision.course_name,
            course_details=json.dumps(course_details)
        )
    )
    
    return {
        "raw_agent_response": result.content,
        "next": "supervisor_agent"
    }

def clarification_agent(state: AgentState) -> Dict:
    """Clarification agent for unclear queries"""
    clarification_prompt = PromptTemplate(
        template=CLARIFICATION_PROMPT,
        input_variables=["query"],
        partial_variables={"format_instructions": clarification_parser.get_format_instructions()}
    )
    
    result = llm.invoke(
        clarification_prompt.format(query=state["query"])
    )
    
    return {
        "raw_agent_response": result.content,
        "next": "supervisor_agent"
    }

def supervisor_agent(state: AgentState) -> Dict:
    """Supervisor agent that generates the final response"""
    chat_history_str = "\n".join([
        f"{'Human' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
        for msg in state.get("chat_history", [])
    ])
    
    supervisor_prompt = PromptTemplate(
        template=SUPERVISOR_PROMPT,
        input_variables=["chat_history", "query", "agent_response"]
    )
    
    result = llm.invoke(
        supervisor_prompt.format(
            chat_history=chat_history_str,
            query=state["query"],
            agent_response=state["raw_agent_response"]
        )
    )
    
    # Parse the agent response
    parsed_response = None
    if state["raw_agent_response"]:
        parsed_response = clean_and_parse_json(state["raw_agent_response"])
    
    return {
        "final_response": result.content,
        "parsed_agent_response": parsed_response,
        "next": "title_generator"
    }

def title_generator(state: AgentState) -> Dict:
    """Generate title for the chat session"""
    # Only generate title if this is the first message
    if not state.get("chat_history") or len(state["chat_history"]) == 0:
        title_prompt = PromptTemplate(
            template=TITLE_PROMPT,
            input_variables=["query"]
        )
        
        result = llm.invoke(
            title_prompt.format(query=state["query"])
        )
        
        return {
            "chat_title": result.content.strip(),
            "next": "save_conversation_agent"  # Updated to match new node name
        }
    
    return {"next": "save_conversation_agent"}  # Updated to match new node name

def save_conversation(state: AgentState) -> Dict:
    """Save the conversation to database"""
    # Your save logic here
    save_chat_message(state["user_id"], state["session_id"], "user", state["query"])
    save_chat_message(state["user_id"], state["session_id"], "assistant", state["final_response"])
    
    if state.get("chat_title"):
        save_chat_session(
            session_id=state["session_id"],
            user_id=state["user_id"],
            title=state["chat_title"]
        )
    
    return {"next": END}

# Create the graph
def create_course_recommendation_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("decision_agent", decision_agent)
    workflow.add_node("recommendation_agent", recommendation_agent)
    workflow.add_node("inquiry_agent", inquiry_agent)
    workflow.add_node("clarification_agent", clarification_agent)
    workflow.add_node("supervisor_agent", supervisor_agent)
    workflow.add_node("title_generator", title_generator)
    workflow.add_node("save_conversation_agent", save_conversation)
    
    # Set entry point
    workflow.set_entry_point("decision_agent")
    
    # Add conditional edges based on decision
    workflow.add_conditional_edges(
        "decision_agent",
        lambda x: x["next"],
        {
            "recommendation": "recommendation_agent",
            "inquiry": "inquiry_agent",
            "clarification_needed": "clarification_agent"
        }
    )
    
    # All agents go to supervisor
    workflow.add_edge("recommendation_agent", "supervisor_agent")
    workflow.add_edge("inquiry_agent", "supervisor_agent")
    workflow.add_edge("clarification_agent", "supervisor_agent")
    
    # Supervisor goes to title generation
    workflow.add_edge("supervisor_agent", "title_generator")
    
    # Title generation goes to save
    workflow.add_edge("title_generator", "save_conversation_agent")
    
    return workflow.compile()

# Generate a random session ID
def generate_session_id(user_id):
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    return f"session_{user_id}_{random_str}"

# Updated CourseRecommenderSystem class
class CourseRecommenderSystem:
    def __init__(self, user_id, db_session):
        self.db = db_session
        self.user_id = user_id
        self.session_id = generate_session_id(user_id)
        self.session_created_at = datetime.now().isoformat()
        self.chat_history = []
        self.graph = create_course_recommendation_graph()
        self.last_decision = None
        self.chat_title = None
        self.all_sessions = {}  # In-memory storage for sessions
    
    def generate_chat_title(self, first_query):
        """Generate a descriptive title for the chat session based on the first query."""
        try:
            title_prompt = PromptTemplate(
                template=TITLE_PROMPT,
                input_variables=["query"]
            )
            
            result = llm.invoke(title_prompt.format(query=first_query))
            title = result.content.strip()
            
            if not title:
                words = first_query.split()
                title = " ".join(words[:4]) + ("..." if len(words) > 4 else "")
            
            return title
        except Exception as e:
            print(f"Error generating title: {e}")
            return f"Course Conversation {datetime.now().strftime('%Y-%m-%d')}"
    
    def save_chat_session(self, title=None):
        """Save the current chat session with metadata including title."""
        chat_history = self.chat_history if self.chat_history else []
        first_query = ""
        
        if chat_history and len(chat_history) > 0:
            first_query = chat_history[0].get("content", "")
        
        if not title and first_query:
            title = self.generate_chat_title(first_query)
        elif not title:
            title = f"Course Conversation {datetime.now().strftime('%Y-%m-%d')}"
        
        session_metadata = {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "title": title,
            "created_at": self.session_created_at,
            "last_active": datetime.now().isoformat(),
            "first_message": first_query
        }
        
        self._save_session_metadata(session_metadata)
        
        return session_metadata
    
    def _save_session_metadata(self, metadata):
        """Save session metadata to database (placeholder implementation)."""
        print(f"Saving session metadata: {json.dumps(metadata, indent=2)}")
        self.all_sessions[metadata['session_id']] = metadata
        
        save_chat_session(
            session_id=metadata['session_id'],
            user_id=metadata['user_id'],
            title=metadata['title'],
            metadata={k: v for k, v in metadata.items() if k not in ['session_id', 'user_id', 'title']}
        )
    
    def list_user_sessions(self, user_id=None):
        """List all chat sessions for a user."""
        target_user = user_id if user_id else self.user_id
        
        user_sessions = [
            session for session in self.all_sessions.values()
            if session['user_id'] == target_user
        ]
        
        external_sessions = get_user_sessions(target_user)
        
        all_sessions = {session['session_id']: session for session in user_sessions + external_sessions}
        
        return list(all_sessions.values())
    
    def process_query(self, query):
        """Process a user query through the graph"""
        # Check if this is the first query (no chat history)
        is_first_query = len(self.chat_history) == 0
        
        # Prepare initial state
        initial_state = {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "query": query,
            "chat_history": self.chat_history,
            "db_session": self.db,
            "next": "decision"
        }
        
        try:
            # Run the graph
            result = self.graph.invoke(initial_state)
            
            # Update chat history
            self.chat_history.append({"role": "user", "content": query})
            self.chat_history.append({"role": "assistant", "content": result["final_response"]})
            
            # If this is the first query, generate and save a title
            if is_first_query and result.get("chat_title"):
                self.chat_title = result["chat_title"]
                self.save_chat_session(title=self.chat_title)
                print(f"Generated chat title: {self.chat_title}")
            
            return (
                result["final_response"],
                result.get("parsed_agent_response"),
                result.get("chat_title", self.chat_title)
            )
            
        except Exception as e:
            print(f"Error processing query: {e}")
            error_message = f"I'm sorry, I encountered an error while processing your request. Please try again with a clearer question about your course needs or career goals."
            
            # Still save the chat history even on error
            self.chat_history.append({"role": "user", "content": query})
            self.chat_history.append({"role": "assistant", "content": error_message})
            
            return error_message, None, self.chat_title

# Session Manager for managing multiple sessions
class SessionManager:
    def __init__(self):
        """Initialize the session manager."""
        self.active_sessions = {}
    
    def get_or_create_session(self, user_id, db_session):
        """Get an existing session or create a new one for a user."""
        if user_id in self.active_sessions:
            return self.active_sessions[user_id]
        
        recommender = CourseRecommenderSystem(user_id, db_session)
        self.active_sessions[user_id] = recommender
        return recommender
    
    def get_session_by_id(self, session_id):
        """Get a specific session by ID."""
        for user_id, session in self.active_sessions.items():
            if session.session_id == session_id:
                return session
        return None
    
    def list_user_sessions(self, user_id):
        """List all sessions for a user."""
        if user_id in self.active_sessions:
            return self.active_sessions[user_id].list_user_sessions(user_id)
        return get_user_sessions(user_id)

# Example usage
if __name__ == "__main__":
    # Create a mock DB session (replace with your actual DB session)
    class MockDBSession:
        pass
    
    db_session = MockDBSession()
    
    # Initialize the session manager
    session_manager = SessionManager()
    
    # Test with different users and query types
    user_id = "student123"
    course_rec_system = session_manager.get_or_create_session(user_id, db_session)
    
    # Example 1: Career goal query
    query = "I want to become a data scientist. What courses should I take?"
    print(f"\nUser: {query}")
    response, agent_response, title = course_rec_system.process_query(query)
    print(f"Response: {response[:200]}...")
    print(f"Chat Title: {title}")
    
    # Example 2: Different user with course inquiry
    user_id2 = "student456"
    course_rec_system2 = session_manager.get_or_create_session(user_id2, db_session)
    
    query2 = "Can you tell me about the Machine Learning course?"
    print(f"\nUser 2: {query2}")
    response2, agent_response2, title2 = course_rec_system2.process_query(query2)
    print(f"Response: {response2[:200]}...")
    print(f"Chat Title: {title2}")
    
    # Example 3: Vague query
    user_id3 = "student789"
    course_rec_system3 = session_manager.get_or_create_session(user_id3, db_session)
    
    query3 = "I'm feeling overwhelmed with all the options. Can you help me?"
    print(f"\nUser 3: {query3}")
    response3, agent_response3, title3 = course_rec_system3.process_query(query3)
    print(f"Response: {response3[:200]}...")
    print(f"Chat Title: {title3}")
    
    # Example 4: Follow-up conversation
    print(f"\nUser: Following up on data science...")
    follow_up_query = "What about machine learning prerequisites?"
    follow_up_response, follow_up_agent, _ = course_rec_system.process_query(follow_up_query)
    print(f"Follow-up Response: {follow_up_response[:200]}...")
    
    # List all sessions
    print("\n=== All Sessions ===")
    all_users = [user_id, user_id2, user_id3]
    for uid in all_users:
        sessions = session_manager.list_user_sessions(uid)
        print(f"\nUser {uid} sessions:")
        for session in sessions:
            print(f"- {session.get('title')} (ID: {session.get('session_id')})")
            print(f"  Created: {session.get('created_at')}")
            print(f"  Last Active: {session.get('last_active', 'N/A')}")