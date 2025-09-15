from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class CompanyData(BaseModel):
    CompanyName_CH: str
    CompanyName_EN: str
    Enterprise_Number: str
    Industry_category: str
    Supplier_evaluation_information: dict
    Phhone_Number: str
    Address: str
    Website: str
    Email: str
    keywords: list
    Score: float = 1.0

class DataStructure(BaseModel):
    company: str
    product: str
    industry: str
    country: str
    document: str
    issue_date: datetime
    expire_date: datetime

#Search
class SearchRequest(BaseModel):
    query_text: str
    filters: str
    top_k: int = 5


# class NumericGap(BaseModel):
#     lead_time: Optional[str] = None
#     quality: Optional[str] = None
#     capacity: Optional[str] = None


class SearchResult(BaseModel):
    company: str
    product: Optional[str] = None
    completeness_score: int
    semantic_score: float
    # numeric_gap: NumericGap
    doc_status: str
    total_score: int




#Feedback
class FeedbackRequest(BaseModel):
    query_id: str
    result_id: str
    action_type: str  # "keep", "reject", "compare"


class FeedbackResponse(BaseModel):
    status: str  # "success" or "failure"
    message: str
