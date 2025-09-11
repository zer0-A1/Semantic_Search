import pandas as pd
import uuid
import json
from fastapi import APIRouter, UploadFile, File, HTTPException
from database.database import Data, VectorDB, async_session, get_session
from database.schemas import UploadFile as UploadFileSchema
router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    if file.content_type not in ["text/csv", "application/json"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only CSV and JSON are allowed.")
    if file.content_type == "text/csv":
        df = pd.read_csv(file.file)
    else:
        df = pd.read_json(file.file)
        