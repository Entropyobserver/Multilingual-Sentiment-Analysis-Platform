from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import pipeline
import logging
import time
import os
import uuid
from typing import Dict, Optional
from functools import lru_cache
from datetime import datetime, timedelta

# Environment configuration
@lru_cache()
def get_settings():
    return {
        "max_requests_per_hour": int(os.getenv("MAX_REQUESTS_PER_HOUR", "100")),
        "cache_duration": int(os.getenv("CACHE_DURATION", "3600")),
        "max_text_length": int(os.getenv("MAX_TEXT_LENGTH", "500")),
        "model_name": os.getenv("MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english")
    }

settings = get_settings()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sentiment Analysis API",
    version="1.0.0",
    description="Transformers-based sentiment analysis API optimized for Azure free tier"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
request_cache: Dict[str, Dict] = {}
request_counts: Dict[str, int] = {}

# Lazy-loaded model with caching
@lru_cache(maxsize=1)
def get_sentiment_model():
    """Lazy load sentiment analysis model with caching"""
    logger.info("Loading sentiment analysis model...")
    model = pipeline(
        "sentiment-analysis",
        model=settings["model_name"],
        device=-1,  # Force CPU usage for Azure free tier
        tokenizer_kwargs={'clean_up_tokenization_spaces': True}
    )
    logger.info("Model loaded successfully!")
    return model

class TextRequest(BaseModel):
    text: str

class AnalysisResponse(BaseModel):
    text: str
    label: str
    score: float
    confidence: str
    cached: bool = False
    request_id: str

@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    """Add unique request ID for tracking and debugging"""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(round(process_time, 4))
    
    logger.info(f"Request {request_id} processed in {process_time:.4f}s")
    return response

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware"""
    client_ip = request.client.host
    current_time = datetime.now()
    
    # Clean expired request counts
    for ip in list(request_counts.keys()):
        if current_time - getattr(request_counts.get(ip, {}), 'timestamp', current_time) > timedelta(hours=1):
            del request_counts[ip]
    
    # Check rate limit
    if request_counts.get(client_ip, 0) >= settings["max_requests_per_hour"]:
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests, please try again later"}
        )
    
    # Update request count
    request_counts[client_ip] = request_counts.get(client_ip, 0) + 1
    
    response = await call_next(request)
    return response

def clean_cache():
    """Clean expired cache entries"""
    current_time = time.time()
    for key in list(request_cache.keys()):
        if current_time - request_cache[key]["timestamp"] > settings["cache_duration"]:
            del request_cache[key]

def get_cached_result(text: str) -> Optional[Dict]:
    """Get cached analysis result"""
    clean_cache()
    cache_key = text.strip().lower()
    cached_item = request_cache.get(cache_key)
    if cached_item and time.time() - cached_item["timestamp"] <= settings["cache_duration"]:
        return cached_item["result"]
    return None

def cache_result(text: str, result: Dict):
    """Cache analysis result"""
    cache_key = text.strip().lower()
    request_cache[cache_key] = {
        "result": result,
        "timestamp": time.time()
    }

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Sentiment Analysis API is running",
        "status": "healthy",
        "docs_url": "/docs",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint for Azure monitoring"""
    try:
        # Test model loading
        model = get_sentiment_model()
        model_status = "loaded"
    except Exception as e:
        logger.error(f"Model health check failed: {e}")
        model_status = "failed"
    
    return {
        "status": "healthy" if model_status == "loaded" else "unhealthy",
        "model_status": model_status,
        "cache_size": len(request_cache),
        "active_requests": len(request_counts),
        "timestamp": datetime.now().isoformat(),
        "settings": {
            "max_requests_per_hour": settings["max_requests_per_hour"],
            "cache_duration": settings["cache_duration"],
            "max_text_length": settings["max_text_length"]
        }
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_sentiment(request: TextRequest, req: Request):
    """Sentiment analysis endpoint"""
    request_id = getattr(req.state, 'request_id', 'unknown')
    
    try:
        # Load model (cached after first call)
        sentiment_analyzer = get_sentiment_model()
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise HTTPException(status_code=503, detail="Model not available, please try again later")
    
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if len(text) > settings["max_text_length"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Text length cannot exceed {settings['max_text_length']} characters"
        )
    
    # Check cache
    cached_result = get_cached_result(text)
    if cached_result:
        logger.info(f"Cache hit for request {request_id}")
        return AnalysisResponse(**cached_result, cached=True, request_id=request_id)
    
    try:
        # Perform sentiment analysis
        result = sentiment_analyzer(text)[0]
        
        # Map labels to Chinese
        label_map = {
            "POSITIVE": "Positive",
            "NEGATIVE": "Negative"
        }
        
        # Calculate confidence level
        confidence_level = "High" if result["score"] > 0.8 else "Medium" if result["score"] > 0.6 else "Low"
        
        response_data = {
            "text": text,
            "label": label_map.get(result["label"], result["label"]),
            "score": round(result["score"], 4),
            "confidence": confidence_level
        }
        
        # Cache result
        cache_result(text, response_data)
        
        logger.info(f"Analysis completed for request {request_id}: {response_data['label']} ({response_data['score']})")
        
        return AnalysisResponse(**response_data, cached=False, request_id=request_id)
    
    except Exception as e:
        logger.error(f"Analysis failed for request {request_id}: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed, please try again")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)