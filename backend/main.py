from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, ConfigDict
from typing import Optional

# Import your existing CourseRecommenderSystem
from course_recommendation_system import CourseRecommenderSystem

# Create FastAPI app
app = FastAPI(
    title="Course Recommendation System",
    description="AI-powered course recommendation and academic guidance API",
    version="1.0.0"
)

# Request model for chat input
class ChatRequest(BaseModel):
    user_id: str
    query: str
    
    # Add model_config for Pydantic v2 compatibility
    model_config = ConfigDict(from_attributes=True)

# Response model for chat output
class ChatResponse(BaseModel):
    response: str
    session_id: Optional[str] = None
    
    # Add model_config for Pydantic v2 compatibility
    model_config = ConfigDict(from_attributes=True)

# Dependency to manage course recommender instances
class CourseRecommenderManager:
    _recommenders = {}

    def get_recommender(self, user_id: str):
        if user_id not in self._recommenders:
            self._recommenders[user_id] = CourseRecommenderSystem(user_id)
        return self._recommenders[user_id]

recommender_manager = CourseRecommenderManager()

@app.post("/chat", response_model=ChatResponse)
async def process_chat(
    request: ChatRequest, 
    recommender_manager: CourseRecommenderManager = Depends(lambda: recommender_manager)
):
    """
    Process a user's chat query and return the AI-generated response.
    
    - user_id: Unique identifier for the user
    - query: User's natural language query about courses or career guidance
    
    Returns an AI-generated response with optional session ID.
    """
    try:
        # Validate input
        if not request.user_id or not request.query:
            raise HTTPException(status_code=400, detail="User ID and query are required")
        
        # Get or create recommender for this user
        recommender = recommender_manager.get_recommender(request.user_id)
        
        # Process the query
        response = recommender.process_query(request.query)
        
        # Return response with session ID
        return ChatResponse(
            response=response, 
            session_id=recommender.session_id
        )
    
    except Exception as e:
        # Log the error (you might want to use a proper logging system)
        print(f"Error processing chat: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the request")

# Additional endpoints for system management (optional)
@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy"}

@app.get("/session/{user_id}")
async def get_user_session(
    user_id: str, 
    recommender_manager: CourseRecommenderManager = Depends(lambda: recommender_manager)
):
    """
    Retrieve the current session ID for a user.
    Useful for tracking conversation context.
    """
    recommender = recommender_manager.get_recommender(user_id)
    return {"session_id": recommender.session_id}

# Optional: Add CORS middleware if needed
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Startup event (optional)
@app.on_event("startup")
async def startup_event():
    print("Course Recommendation System API is starting up...")

# Shutdown event (optional)
@app.on_event("shutdown")
async def shutdown_event():
    print("Course Recommendation System API is shutting down...")

# If you want to run this directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)