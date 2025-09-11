from fastapi import FastAPI
import asyncio
import logging
from database.database import init_db

# #Import routers
# from api.upload import router as upload_router
# from api.search import router as search_router
# from api.feedback import router as feedback_router

app = FastAPI()
# app.include_router(upload_router, prefix="/api/upload")
# app.include_router(search_router, prefix="/api/search")
# app.include_router(feedback_router, prefix="/api/feedback")

@app.on_event("startup")
async def on_startup():
    try:
        await init_db()
    except Exception as e:
        logging.error(f"Database Initialization failed: {e}")