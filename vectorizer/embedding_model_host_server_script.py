import os
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global model variable
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    
    try:
        logger.info("Loading SentenceTransformer model...")
        # Default model - can be overridden by environment variable
        model_id = os.environ.get("MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Using device: {device}")
        logger.info(f"Loading model: {model_id}")
        
        model = SentenceTransformer(model_id, device=device)
        logger.info("Model loaded successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise RuntimeError(f"Error loading model: {str(e)}")
    
    finally:
        logger.info("Cleaning up resources...")
        if model:
            del model
        torch.cuda.empty_cache()

# Initialize FastAPI app with lifespan manager
app = FastAPI(
    title="Text Embedding API",
    description="API for generating text embeddings using SentenceTransformer",
    version="1.0.0",
    lifespan=lifespan
)

class EmbeddingRequest(BaseModel):
    texts: List[str]
    batch_size: Optional[int] = 32

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model_id: str
    dimensions: int

@app.post("/embed", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    """
    Generate embeddings for a list of text inputs.
    
    Parameters:
    - texts: List of text strings to encode
    - batch_size: Number of texts to process at once (optional, default: 32)
    
    Returns:
    - embeddings: List of embedding vectors
    - model_id: Name of the model used
    - dimensions: Dimensionality of the embeddings
    """
    global model
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info(f"Processing {len(request.texts)} texts with batch size {request.batch_size}")
        
        # Encode the texts
        embeddings = model.encode(
            request.texts, 
            batch_size=request.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Convert to Python list for JSON serialization
        embeddings_list = embeddings.tolist()
        
        dimensions = len(embeddings_list[0]) if embeddings_list else 0
        logger.info(f"Generated embeddings with dimensions: {dimensions}")
        
        return EmbeddingResponse(
            embeddings=embeddings_list,
            model_id=model.model_card_data["base_model"],
            dimensions=dimensions
        )
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Check if the API is healthy and the model is loaded.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model": model.model_card_data["base_model"],
        "device": next(model.parameters()).device.type
    }

@app.get("/info")
async def model_info():
    """
    Get information about the loaded model.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_id": model.model_card_data["base_model"],
        "embedding_dimension": model.get_sentence_embedding_dimension(),
        "max_seq_length": model.max_seq_length,
        "device": next(model.parameters()).device.type
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8001))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)