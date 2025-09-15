from fastapi import FastAPI
import asyncio
import logging
import os
from database.database import init_db

# Import routers
from api.upload import router as upload_router
from api.search import router as search_router
from api.feedback import router as feedback_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app with production settings
app = FastAPI(
    title="Semantic Search API",
    description="AI-powered semantic search for company data",
    version="1.0.0",
    docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
    redoc_url="/redoc" if os.getenv("ENVIRONMENT") != "production" else None,
)

# Include routers
app.include_router(upload_router, prefix="/api")
app.include_router(search_router, prefix="/api")
app.include_router(feedback_router, prefix="/api")


@app.on_event("startup")
async def on_startup():
    """Initialize database on startup"""
    try:
        logger.info("Starting database initialization...")
        await init_db()
        logger.info("Database initialization completed successfully")
    except Exception as e:
        logger.error(f"Database Initialization failed: {e}")
        raise e


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Semantic Search API is running", "status": "healthy"}


@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy",
        "service": "semantic_search_api",
        "version": "1.0.0"
    }
