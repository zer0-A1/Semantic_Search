from pydantic import BaseModel
from datetime import datetime

class UploadFile(BaseModel):
    "filename": str
    "type": str
    "score": float
    
class SearchQuery(BaseModel):
    "query": str
    "top_k": int
    "filter": dict = None
    
class DataStructure(BaseModel):
    "company":str
    "product":str
    "industry":str
    "country":str
    "document":str
    "issue_date":datetime
    "expire_date":datetime
    