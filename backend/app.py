import os
import logging
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine, func, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import create_engine, func, select, text, desc, or_, and_, exists, column
from sqlalchemy.sql.expression import literal, except_
from passlib.context import CryptContext
import sqlalchemy as sa
from contextlib import asynccontextmanager
from typing import Optional
from postgresql_model import Login, StudentProfile, CourseDetails, CompletedCourse, MyCourseList, CourseTrends, ChatSession, ChatMessage, CareerPath
from postgresql_model import CourseReview, RecommendationHistory
from postgresql_model import Base
import requests
from vectorizer_models import VectorizerClient
from anthropic_client import AnthropicClient
from prompt_creator import PromptGenerator
import re
import json
from sqlalchemy import select, func, and_, desc, cast, text
from sqlalchemy.sql.expression import or_
from sqlalchemy.dialects.postgresql import ARRAY
from typing import List, Dict, Any, Optional
import requests
from pgvector.sqlalchemy import Vector
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

from course_recommendation_system import CourseRecommenderSystem

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
DB_HOST     = os.environ.get("DB_HOST", "")
DB_NAME     = os.environ.get("DB_NAME", "")
DB_USER     = os.environ.get("DB_USER", "")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_PORT     = int(os.environ.get("DB_PORT", 5432))


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PROMPTS_DIR = os.path.join(SCRIPT_DIR, "prompts")
DECISION_PROMPT_DIR = os.path.join(PROMPTS_DIR, "decision_task")
RECOMMEND_PROMPT_DIR = os.path.join(PROMPTS_DIR, "recommend_task")

decision_base_prompt_file =  os.path.join(DECISION_PROMPT_DIR, "base_prompt.txt")
decision_few_shot_examples_file = os.path.join(DECISION_PROMPT_DIR, "few_shots.txt")

# DB Configuration
db_config = {
    "host": "localhost",
    "port": 5432,
    "database": "smart_search_course_recommendation",
    "user": "postgres",
    "password": "mz7zdz123"
}

# SQLAlchemy setup
DATABASE_URL = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
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


# Response model for chat output
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


# # Setup database connection
# DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
# engine = create_engine(DATABASE_URL)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


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
            anthropic_model = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
            if anthropic_api_key:
                anthropic_client = AnthropicClient(api_key=anthropic_api_key, model=anthropic_model)
                logger.info(f"Successfully initialized Anthropic client with model: {anthropic_model}")
            else:
                logger.warning("ANTHROPIC_API_KEY not found in environment variables")
                logger.warning("RAG functionality will not be available")
        except Exception as e:
            logger.warning(f"Could not initialize Anthropic client: {e}")
            logger.warning("RAG functionality will not be available")
        
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

# Add this after creating your FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

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
    """
    Register a new user account
    """
    logger.info(f"Processing signup request for user_id: {user.user_id}")
    
    # Check if user already exists
    existing_user = db.query(Login).filter(
        (Login.user_id == user.user_id) | (Login.email == user.email)
    ).first()
    
    if existing_user:
        if existing_user.user_id == user.user_id:
            logger.warning(f"Signup failed: User ID {user.user_id} already exists")
            raise HTTPException(
                status_code=400,
                detail="User ID already registered"
            )
        else:
            logger.warning(f"Signup failed: Email {user.email} already exists")
            raise HTTPException(
                status_code=400,
                detail="Email already registered"
            )
    
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
        raise HTTPException(
            status_code=500,
            detail=f"Error creating user: {str(e)}"
        )

@app.post("/login", response_model=UserResponse)
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """
    Authenticate user credentials
    """
    logger.info(f"Processing login request for user_id: {credentials.user_id}")
    
    # Find user
    user = db.query(Login).filter(Login.user_id == credentials.user_id).first()
    
    # Verify credentials
    if not user or not verify_password(credentials.password, user.password):
        logger.warning(f"Login failed for user_id: {credentials.user_id}")
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )
    
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
    """
    Fetch all courses from the database
    """
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
        return {"courses": course_list[:200]}
    
    except Exception as e:
        logger.error(f"Error fetching courses: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching courses: {str(e)}"
        )


# Dependency to manage course recommender instances
class CourseRecommenderManager:
    _recommenders = {}

    def get_recommender(self, user_id: str, db: Session = Depends(get_db)):
        if user_id not in self._recommenders:
            self._recommenders[user_id] = CourseRecommenderSystem(user_id, db)
        return self._recommenders[user_id]

recommender_manager = CourseRecommenderManager()
@app.post("/chat")
async def process_chat(request: ChatRequest, db: Session = Depends(get_db), 
                       recommender_manager: CourseRecommenderManager = Depends(lambda: recommender_manager)):
    """
    Process a user query with Anthropic and return the generated response
    """
    logger.info(f"Processing chat request for user_id: {request.user_id}")
    
    try:

        if not request.user_id or not request.query:
            raise HTTPException(status_code=400, details="User ID and query are required")
        
        # Get or create recommender for this user
        recommender = recommender_manager.get_recommender(request.user_id)
        
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
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        db.rollback()
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}"
        )
    

@app.get("/trends/{course_id}", response_model=List[CourseTrendResponse])
def get_course_trends(course_id: str, db: Session = Depends(get_db)):
    trends = db.query(CourseTrends).filter(CourseTrends.course_id == course_id).order_by(CourseTrends.year.desc()).all()

    if not trends:
        raise HTTPException(status_code=404, detail="No trends found for this course")

    return trends

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
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)