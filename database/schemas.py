from pydantic import BaseModel
from datetime import datetime
from typing import Optional

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
