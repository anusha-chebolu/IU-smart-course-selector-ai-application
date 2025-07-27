import os
import logging
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import create_engine, func, select, text, desc, or_, and_, exists, column, cast
from sqlalchemy.sql.expression import literal, except_, or_
from passlib.context import CryptContext
import sqlalchemy as sa
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from postgresql_model import Login, StudentProfile, CourseDetails, CompletedCourse, MyCourseList, CourseTrends, ChatSession, ChatMessage, CareerPath
from postgresql_model import CourseReview, RecommendationHistory
from postgresql_model import Base
from sqlalchemy.dialects.postgresql import ARRAY
from vectorizer_models import VectorizerClient
from anthropic_client import AnthropicClient
from prompt_creator import PromptGenerator
from pgvector.sqlalchemy import Vector
from fastapi.middleware.cors import CORSMiddleware
import re
import json
import requests


from dotenv import load_dotenv

# Import the new LangGraph-based system
from course_recommendation_system_langgraph import CourseRecommenderSystem, SessionManager

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global variables
embedding_api_url       = os.environ.get("EMBEDDING_API_URL", "http://127.0.0.1:8001/embed")
anthropic_key           = os.environ.get("ANTHROPIC_API_KEY", "")
anthropic_client        = None

# Database connection
DB_HOST                 = os.environ.get("DB_HOST", "localhost")
DB_NAME                 = os.environ.get("DB_NAME", "smart_search_course_recommendation")
DB_USER                 = os.environ.get("DB_USER", "postgres")
DB_PASSWORD             = os.environ.get("DB_PASSWORD", "mz7zdz123")
DB_PORT                 = int(os.environ.get("DB_PORT", 5432))

# SQLAlchemy setup
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

# Define Request Pydantic models
class UserCreate(BaseModel):
    user_id: str
    first_name: str
    last_name: str
    email: str
    password: str

class UserLogin(BaseModel):
    user_id: str
    password: str

class ChatRequest(BaseModel):
    user_id: str
    query: str
    chat_history: Optional[str] = None
    session_id: Optional[str] = None

# Define Response Pydantic models
class UserResponse(BaseModel):
    user_id: str
    first_name: str
    last_name: str
    email: str

class ChatResponse(BaseModel):
    response: str
    json_response: Optional[Dict] = None
    chat_title: Optional[str] = None
    session_id: Optional[str] = None

class CourseTrendResponse(BaseModel):
    year: int
    slots_filled: int
    total_slots: int
    avg_rating: float | None
    slots_filled_time: int | None
    avg_gpa: float | None
    avg_hours_spent: float | None

class ChatSessionResponse(BaseModel):
    session_id: str
    user_id: str
    title: str
    created_at: str
    last_active: Optional[str] = None

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def extract_json_string(response_text):
    json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
    
    if json_match:
        json_string = json_match.group(1)
        try:
            json.loads(json_string)
            return json_string, None
        except json.JSONDecodeError as e:
            return None, f"Error: Extracted text is not valid JSON: {str(e)}"
    else:
        return None, "Error: No JSON found in the response"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and clean up resources"""
    global embedding_api_url, anthropic_client
    try:
        try:
            # Test database connection
            logger.info("Connecting to database...")
            db = SessionLocal()
            db.execute(text("SELECT 1"))
            db.close()
            logger.info("Successfully connected to database")
        except Exception as e:
            logger.warning(f"Could not connect to postgresql DB: {e}")
            logger.warning("The service will start, but postgresql DB connection must be available when processing sql queries")
        
        
        # Test connection to embedding API
        embedding_api_url = os.environ.get("EMBEDDING_API_URL", "http://localhost:8001/embed")
        try:
            response = requests.post(
                embedding_api_url,
                json={"texts": ["Test connection to embedding API"]},
                timeout=5
            )
            response.raise_for_status()
            logger.info("Successfully connected to embedding API")
        except Exception as e:
            logger.warning(f"Could not connect to embedding API: {e}")
            logger.warning("The service will start, but embedding API must be available when processing queries")
        
        # Initialize Anthropic client
        try:
            anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
            anthropic_model = os.environ.get("ANTHROPIC_MODEL", "claude-3-7-sonnet-20250219")
            if anthropic_api_key:
                anthropic_client = AnthropicClient(api_key=anthropic_api_key, model=anthropic_model)
                logger.info(f"Successfully initialized Anthropic client with model: {anthropic_model}")
            else:
                logger.warning("ANTHROPIC_API_KEY not found in environment variables")
        except Exception as e:
            logger.warning(f"Could not initialize Anthropic client: {e}")
        
        yield
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise
    finally:
        logger.info("Cleaning up resources...")

# Initialize FastAPI app
app = FastAPI(
    title="Smart Course Selector API",
    description="API for student course selection and recommendations",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global session manager
session_manager = SessionManager()

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Password utilities
def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

@app.post("/signup", response_model=UserResponse)
async def signup(user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user account"""
    logger.info(f"Processing signup request for user_id: {user.user_id}")
    
    # Check if user already exists
    existing_user = db.query(Login).filter(
        (Login.user_id == user.user_id) | (Login.email == user.email)
    ).first()
    
    if existing_user:
        if existing_user.user_id == user.user_id:
            logger.warning(f"Signup failed: User ID {user.user_id} already exists")
            raise HTTPException(status_code=400, detail="User ID already registered")
        else:
            logger.warning(f"Signup failed: Email {user.email} already exists")
            raise HTTPException(status_code=400, detail="Email already registered")
    
    # Hash password
    hashed_password = get_password_hash(user.password)
    
    try:
        # Create new user in login table
        new_user = Login(
            user_id=user.user_id,
            first_name=user.first_name,
            last_name=user.last_name,
            email=user.email,
            password=hashed_password
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        logger.info(f"User {user.user_id} successfully created")
        
        return {
            "user_id": new_user.user_id,
            "first_name": new_user.first_name,
            "last_name": new_user.last_name,
            "email": new_user.email
        }
    
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating user: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating user: {str(e)}")

@app.post("/login", response_model=UserResponse)
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """Authenticate user credentials"""
    logger.info(f"Processing login request for user_id: {credentials.user_id}")
    
    # Find user
    user = db.query(Login).filter(Login.user_id == credentials.user_id).first()
    
    # Verify credentials
    if not user or not verify_password(credentials.password, user.password):
        logger.warning(f"Login failed for user_id: {credentials.user_id}")
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Update last login timestamp
    user.last_login = func.now()
    db.commit()
    
    logger.info(f"User {credentials.user_id} successfully authenticated")
    
    return {
        "user_id": user.user_id,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "email": user.email,
        "created_at": user.created_at
    }

@app.get("/course_catalog")
async def get_course_catalog(db: Session = Depends(get_db)):
    """Fetch all courses from the database"""
    logger.info("Processing request to fetch all courses")
    
    try:
        # Query all courses from the CourseDetails table
        courses = db.query(CourseDetails).all()
        
        # Convert SQLAlchemy objects to dictionaries
        course_list = []
        for course in courses:
            course_dict = {
                "course_id": course.course_id,
                "course_name": course.course_name,
                "department": course.department,
                "min_credits": course.min_credits,
                "max_credits": course.max_credits,
                "prerequisites": course.prerequisites,
                "offered_semester": course.offered_semester,
                "course_title": course.course_title,
                "course_description": course.course_description,
                "course_details": course.course_details
            }
            course_list.append(course_dict)
        
        logger.info(f"Successfully retrieved {len(course_list)} courses")
        return {"courses": course_list[:100]}
    
    except Exception as e:
        logger.error(f"Error fetching courses: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching courses: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def process_chat(request: ChatRequest, db: Session = Depends(get_db)):
    """Process a user query with the LangGraph-based system"""
    logger.info(f"Processing chat request for user_id: {request.user_id}")
    
    try:
        if not request.user_id or not request.query:
            raise HTTPException(status_code=400, detail="User ID and query are required")
        
        # Get or create session
        if request.session_id:
            # Try to get existing session
            recommender = session_manager.get_session_by_id(request.session_id)
            if not recommender:
                # If session not found, create new one
                recommender = session_manager.get_or_create_session(request.user_id, db)
        else:
            # Create new session
            print("creating session id")
            recommender = session_manager.get_or_create_session(request.user_id, db)
        
        print("Processing User query")
        # Process the query
        response, agent_response, chat_title = recommender.process_query(request.query)
        
        # Return response with session ID
        return ChatResponse(
            response=response,
            json_response=agent_response,
            chat_title=chat_title,
            session_id=recommender.session_id
        )
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

@app.get("/chat/sessions/{user_id}", response_model=List[ChatSessionResponse])
async def get_user_sessions(user_id: str, db: Session = Depends(get_db)):
    """Get all chat sessions for a user"""
    logger.info(f"Fetching sessions for user_id: {user_id}")
    
    try:
        # Get sessions from session manager
        sessions = session_manager.list_user_sessions(user_id)
        
        # Also get sessions from database
        db_sessions = db.query(ChatSession).filter(ChatSession.user_id == user_id).all()
        
        # Merge and deduplicate sessions
        all_sessions = {session['session_id']: session for session in sessions}
        
        for db_session in db_sessions:
            if db_session.session_id not in all_sessions:
                all_sessions[db_session.session_id] = {
                    "session_id": db_session.session_id,
                    "user_id": db_session.user_id,
                    "title": db_session.title,
                    "created_at": db_session.created_at.isoformat(),
                    "last_active": db_session.last_active.isoformat() if db_session.last_active else None
                }
        
        return list(all_sessions.values())
    
    except Exception as e:
        logger.error(f"Error fetching sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching sessions: {str(e)}")

@app.get("/chat/messages/{session_id}")
async def get_session_messages(session_id: str, db: Session = Depends(get_db)):
    """Get all messages for a specific chat session"""
    logger.info(f"Fetching messages for session_id: {session_id}")
    
    try:
        messages = db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).order_by(ChatMessage.created_at).all()
        
        return {
            "session_id": session_id,
            "messages": [
                {
                    "message_id": msg.message_id,
                    "role": msg.role,
                    "content": msg.content,
                    "created_at": msg.created_at.isoformat()
                }
                for msg in messages
            ]
        }
    
    except Exception as e:
        logger.error(f"Error fetching messages: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching messages: {str(e)}")

@app.get("/trends/{course_id}")
async def get_course_trends(course_id: str, db: Session = Depends(get_db)):
    """
    Fetch trend data for a specific course and generate an analysis
    """
    logger.info(f"Processing request to fetch trend data for course_id: {course_id}")
    
    try:
        global anthropic_client
        # Query the course to get its name
        course = db.query(CourseDetails).filter(CourseDetails.course_id == course_id).first()
        
        if not course:
            logger.warning(f"Course with ID {course_id} not found")
            raise HTTPException(
                status_code=404,
                detail=f"Course with ID {course_id} not found"
            )
        
        # Query the trends data
        trends = db.query(CourseTrends).filter(CourseTrends.course_id == course_id).order_by(CourseTrends.year.desc()).all()
        
        if not trends:
            logger.warning(f"No trend data found for course {course_id}")
            raise HTTPException(
                status_code=404,
                detail="No trends found for this course"
            )
        
        # Convert trends to a list of dictionaries for easier handling
        trends_data = []
        for trend in trends:
            trend_dict = {
                "year": trend.year,
                "slots_filled": trend.slots_filled,
                "total_slots": trend.total_slots,
                "avg_rating": trend.avg_rating if trend.avg_rating else None,
                "slots_filled_time": trend.slots_filled_time if trend.slots_filled_time else None,
                "avg_gpa": trend.avg_gpa if trend.avg_gpa else None,
                "avg_hours_spent": trend.avg_hours_spent if trend.avg_hours_spent else None
            }
            trends_data.append(trend_dict)
        
        print(trends_data)
        # Generate trend analysis using Anthropic
        if anthropic_client:
            try:
                # Create prompt for Anthropic
                prompt = f"""
                Analyze the following trend data for the course '{course.course_name}' (ID: {course_id}):
                
                {json.dumps(trends_data, indent=2)}
                
                Please provide a brief analysis of the trends over time considering:
                1. Enrollment trends (slots_filled vs total_slots)
                2. Student performance (avg_gpa)
                3. Course difficulty (avg_hours_spent)
                4. Student satisfaction (avg_rating)
                
                Focus on identifying patterns, significant changes, and potential insights that would be useful for students considering this course.
                Keep your analysis concise and insightful.
                """
                
                print("calling prompt")
                # Call Anthropic API
                analysis_response = anthropic_client.send_message(prompt)
                trend_analysis = analysis_response.strip()
                
                logger.info(f"Successfully generated trend analysis for course {course_id}")
            except Exception as e:
                logger.error(f"Error generating trend analysis: {str(e)}")
                trend_analysis = "Unable to generate trend analysis at this time."
        else:
            logger.warning("Anthropic client not initialized, skipping trend analysis")
            trend_analysis = "Trend analysis not available."
        
        # Return both the raw trend data and the analysis
        return {
            "course_id": course_id,
            "course_name": course.course_name,
            "trends": trends_data,
            "trend_analysis": trend_analysis
        }
    
    except Exception as e:
        logger.error(f"Error fetching course trends: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching course trends: {str(e)}"
        )

@app.get("/course_detail/{course_id}")
async def get_course_detail(course_id: str, db: Session = Depends(get_db)):
    """
    Fetch details for a specific course by course_id
    """
    logger.info(f"Processing request to fetch course details for course_id: {course_id}")
    
    try:
        # Query the specific course from the CourseDetails table
        course = db.query(CourseDetails).filter(CourseDetails.course_id == course_id).first()
        
        if not course:
            logger.warning(f"Course with ID {course_id} not found")
            raise HTTPException(
                status_code=404,
                detail=f"Course with ID {course_id} not found"
            )
        
        
        # Convert SQLAlchemy object to dictionary
        course_dict = {
            "course_id": course.course_id,
            "course_name": course.course_name,
            "department": course.department,
            "min_credits": course.min_credits,
            "max_credits": course.max_credits,
            "prerequisites": course.prerequisites,
            "offered_semester": course.offered_semester,
            "course_title": course.course_title,
            "course_description": course.course_description,
            "course_details": course.course_details
        }
        
        logger.info(f"Successfully retrieved details for course {course_id}")
        return course_dict
    
    except Exception as e:
        logger.error(f"Error fetching course details: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching course details: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Check if the API is healthy and database is connected"""
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)