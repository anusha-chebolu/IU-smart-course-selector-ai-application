from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from typing import List, Optional, Dict, Any
import json
import os
from dotenv import load_dotenv
import random
import string
from util import get_course_recommendations
import re
from datetime import datetime

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

# Configure LLM
llm = ChatAnthropic(
    model_name="claude-3-7-sonnet-20250219", 
    temperature=0.2,
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=8192
)

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

# Title generation prompt
TITLE_PROMPT = """
Generate a short, descriptive title (5 words or less) for a chat session about course recommendations that starts with this query:

User Query: {query}

Title:
"""

# Initialize output parsers
decision_parser         = PydanticOutputParser(pydantic_object=DecisionOutput)
recommendation_parser   = PydanticOutputParser(pydantic_object=RecommendationOutput)
inquiry_parser          = PydanticOutputParser(pydantic_object=GeneralInquiryOutput)
clarification_parser    = PydanticOutputParser(pydantic_object=ClarificationOutput)

# Initialize agent chains
decision_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        template=DECISION_PROMPT,
        input_variables=["query"],
        partial_variables={"format_instructions": decision_parser.get_format_instructions()}
    ),
    output_key="decision"
)

recommendation_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        template=RECOMMENDATION_PROMPT,
        input_variables=["career_goal", "courses"],
        partial_variables={"format_instructions": recommendation_parser.get_format_instructions()}
    ),
    output_key="recommendations"
)

inquiry_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        template=INQUIRY_PROMPT,
        input_variables=["course_name", "course_details"],
        partial_variables={"format_instructions": inquiry_parser.get_format_instructions()}
    ),
    output_key="course_info"
)

clarification_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        template=CLARIFICATION_PROMPT,
        input_variables=["query"],
        partial_variables={"format_instructions": clarification_parser.get_format_instructions()}
    ),
    output_key="clarification"
)

supervisor_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        template=SUPERVISOR_PROMPT,
        input_variables=["chat_history", "query", "agent_response"]
    ),
    output_key="final_response"
)

# Title generation chain
title_chain = LLMChain(
    llm=ChatAnthropic(
        model_name="claude-3-7-sonnet-20250219",
        temperature=0.7,  # Higher temperature for more creative titles
        max_tokens=30,    # Keep titles concise
        api_key=os.getenv("ANTHROPIC_API_KEY")
    ),
    prompt=PromptTemplate(
        template=TITLE_PROMPT,
        input_variables=["query"]
    )
)

def save_chat_message(user_id, session_id, message_type, content):
    """Save chat message to database"""
    # This is a placeholder - will be replaced with actual database query
    print(f"Saving {message_type} message for user {user_id} in session {session_id}")
    return True

def save_chat_session(session_id, user_id, title, metadata=None):
    """Save chat session metadata to database"""
    # This is a placeholder - will be replaced with actual database query
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
    # This is a placeholder - will be replaced with actual database query
    return []

# Generate a random session ID
def generate_session_id(user_id):
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    return f"session_{user_id}_{random_str}"

# Main agent orchestration
class CourseRecommenderSystem:
    def __init__(self, user_id, db_session):
        self.db = db_session
        self.user_id = user_id
        self.session_id = generate_session_id(user_id)
        self.session_created_at = datetime.now().isoformat()
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        self.last_decision = None
        self.chat_title = None
        self.all_sessions = {}  # In-memory storage for sessions
    
    def generate_chat_title(self, first_query):
        """
        Generate a descriptive title for the chat session based on the first query.
        
        Args:
            first_query: The user's first message in the conversation
            
        Returns:
            A descriptive title for the chat session
        """
        try:
            # Generate the title
            title = title_chain.run(query=first_query).strip()
            
            # Fallback if generation fails or returns empty
            if not title:
                # Extract keywords from the query
                words = first_query.split()
                title = " ".join(words[:4]) + ("..." if len(words) > 4 else "")
            
            return title
        except Exception as e:
            print(f"Error generating title: {e}")
            # Fallback to default title with timestamp
            return f"Course Conversation {datetime.now().strftime('%Y-%m-%d')}"
    
    def save_chat_session(self, title=None):
        """
        Save the current chat session with metadata including title.
        
        Args:
            title: Optional custom title, if None will be generated from first query
            
        Returns:
            Dictionary with session metadata
        """
        # Get the first message if available
        chat_history = self.memory.load_memory_variables({}).get("chat_history", "")
        messages = chat_history.split("\n")
        first_query = ""
        
        # Extract the first user query from chat history
        for message in messages:
            if message.startswith("Human:"):
                first_query = message.replace("Human:", "").strip()
                break
        
        # Generate title if not provided
        if not title and first_query:
            title = self.generate_chat_title(first_query)
        elif not title:
            # Fallback if no title and no first query
            title = f"Course Conversation {datetime.now().strftime('%Y-%m-%d')}"
        
        # Create session metadata
        session_metadata = {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "title": title,
            "created_at": self.session_created_at,
            "last_active": datetime.now().isoformat(),
            "first_message": first_query
        }
        
        # Save to database (placeholder)
        self._save_session_metadata(session_metadata)
        
        return session_metadata
    
    def _save_session_metadata(self, metadata):
        """
        Save session metadata to database (placeholder implementation).
        
        Args:
            metadata: Session metadata dictionary
        """
        # This is a placeholder - replace with your actual database code
        print(f"Saving session metadata: {json.dumps(metadata, indent=2)}")
        
        # Add to internal storage
        self.all_sessions[metadata['session_id']] = metadata
        
        # Call the external save function (would be a database call)
        save_chat_session(
            session_id=metadata['session_id'],
            user_id=metadata['user_id'],
            title=metadata['title'],
            metadata={k: v for k, v in metadata.items() if k not in ['session_id', 'user_id', 'title']}
        )
    
    def list_user_sessions(self, user_id=None):
        """
        List all chat sessions for a user.
        
        Args:
            user_id: Optional user ID, defaults to current user
            
        Returns:
            List of session metadata dictionaries
        """
        target_user = user_id if user_id else self.user_id
        
        # First check internal storage
        user_sessions = [
            session for session in self.all_sessions.values()
            if session['user_id'] == target_user
        ]
        
        # Then check external storage (database)
        external_sessions = get_user_sessions(target_user)
        
        # Combine and deduplicate (based on session_id)
        all_sessions = {session['session_id']: session for session in user_sessions + external_sessions}
        
        return list(all_sessions.values())
    
    def process_query(self, query):
        # Check if this is the first query (no chat history)
        is_first_query = len(self.memory.load_memory_variables({}).get("chat_history", "")) == 0
        
        # Save user message
        save_chat_message(self.user_id, self.session_id, "user", query)
        
        # Step 1: Decision Agent determines intent
        decision_result = decision_chain.run(query=query)
        self.last_decision = decision_result
        
        try:
            decision = decision_parser.parse(decision_result)
            print(f"Decision: {decision.action}")
            agent_response = None
            # Step 2: Route to appropriate agent based on intent
            if decision.action == "recommendation":
                # Get available courses relevant to career goal
                print("calling courses fetched")
                courses = get_course_recommendations(self.user_id, decision.career_goal, self.db)
                print("courses fetched")
                print(courses)
                # Run recommendation agent
                recommendation_result = recommendation_chain.run(
                    career_goal=decision.career_goal,
                    courses=json.dumps(courses)
                )
                agent_response = recommendation_result
                
            elif decision.action == "inquiry":
                # Get course details
                course_details = get_course_recommendations(self.user_id, decision.course_name, self.db)
                
                # Run inquiry agent
                inquiry_result = inquiry_chain.run(
                    course_name=decision.course_name,
                    course_details=json.dumps(course_details)
                )
                agent_response = inquiry_result
                
            elif decision.action == "clarification_needed":
                # Run clarification agent
                clarification_result = clarification_chain.run(query=decision.original_query)
                agent_response = clarification_result
            
            # Step 3: Supervisor Agent generates final response
            chat_history = self.memory.load_memory_variables({})["chat_history"]
            final_response = supervisor_chain.run(
                chat_history=chat_history,
                query=query,
                agent_response=agent_response
            )
            
            # Update memory with the interaction
            self.memory.save_context(
                {"input": query},
                {"output": final_response}
            )
            print(type(agent_response))
            print(agent_response)

            if agent_response:
                agent_response = clean_and_parse_json(agent_response)

            print(type(agent_response))
            
            # Save assistant message
            save_chat_message(self.user_id, self.session_id, "assistant", final_response)

            # If this is the first query, generate and save a title
            if is_first_query:
                self.chat_title = self.generate_chat_title(query)
                self.save_chat_session(title=self.chat_title)
                print(f"Generated chat title: {self.chat_title}")
            
            
            return final_response, agent_response, self.chat_title
            
        except Exception as e:
            print(f"Error processing query: {e}")
            print(f"Raw decision result: {decision_result}")
            return f"I'm sorry, I encountered an error while processing your request. Please try again with a clearer question about your course needs or career goals."

# Session Manager for managing multiple sessions
class SessionManager:
    def __init__(self):
        """Initialize the session manager."""
        self.active_sessions = {}
    
    def get_or_create_session(self, user_id, db_session):
        """
        Get an existing session or create a new one for a user.
        
        Args:
            user_id: User identifier
            db_session: Database session
            
        Returns:
            A CourseRecommenderSystem instance
        """
        if user_id in self.active_sessions:
            return self.active_sessions[user_id]
        
        # Create new session
        recommender = CourseRecommenderSystem(user_id, db_session)
        self.active_sessions[user_id] = recommender
        return recommender
    
    def get_session_by_id(self, session_id):
        """
        Get a specific session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            CourseRecommenderSystem instance or None if not found
        """
        # Check all active sessions
        for user_id, session in self.active_sessions.items():
            if session.session_id == session_id:
                return session
        
        # If not found, could potentially load from database
        return None
    
    def list_user_sessions(self, user_id):
        """
        List all sessions for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of session metadata dictionaries
        """
        # If user has an active session, use its list method
        if user_id in self.active_sessions:
            return self.active_sessions[user_id].list_user_sessions(user_id)
        
        # Otherwise query from storage directly
        return get_user_sessions(user_id)

# # Example usage
# if __name__ == "__main__":
#     # Create a mock DB session (replace with your actual DB session)
#     class MockDBSession:
#         pass
    
#     db_session = MockDBSession()
    
#     # Initialize the session manager
#     session_manager = SessionManager()
    
#     # Test with different users and query types
#     user_id = "student123"
#     course_rec_system = session_manager.get_or_create_session(user_id, db_session)
    
#     # Example 1: Career goal query
#     query = "I want to become a data scientist. What courses should I take?"
#     print(f"\nUser: {query}")
#     response, agent_response, title = course_rec_system.process_query(query)
#     print(f"Response: {response[:100]}...")
#     print(f"Chat Title: {title}")
    
#     # Example 2: Different user with course inquiry
#     user_id2 = "student456"
#     course_rec_system2 = session_manager.get_or_create_session(user_id2, db_session)
    
#     query2 = "Can you tell me about the Machine Learning course?"
#     print(f"\nUser 2: {query2}")
#     response2, agent_response2, title2 = course_rec_system2.process_query(query2)
#     print(f"Response: {response2[:100]}...")
#     print(f"Chat Title: {title2}")
    
#     # Example 3: Vague query
#     user_id3 = "student789"
#     course_rec_system3 = session_manager.get_or_create_session(user_id3, db_session)
    
#     query3 = "I'm feeling overwhelmed with all the options. Can you help me?"
#     print(f"\nUser 3: {query3}")
#     response3, agent_response3, title3 = course_rec_system3.process_query(query3)
#     print(f"Response: {response3[:100]}...")
#     print(f"Chat Title: {title3}")
    
#     # List all sessions
#     print("\nAll Sessions:")
#     all_users = [user_id, user_id2, user_id3]
#     for uid in all_users:
#         sessions = session_manager.list_user_sessions(uid)
#         print(f"User {uid} sessions:")
#         for session in sessions:
#             print(f"- {session.get('title')} (ID: {session.get('session_id')})")