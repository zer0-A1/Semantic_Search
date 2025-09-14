from pydantic import BaseModel
from datetime import datetime


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


class SearchQuery(BaseModel):
    query: str
    top_k: int
    filter: dict = None


class DataStructure(BaseModel):
    company: str
    product: str
    industry: str
    country: str
    document: str
    issue_date: datetime
    expire_date: datetime
