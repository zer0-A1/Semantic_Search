import pandas as pd
import uuid
import json
from fastapi import APIRouter, UploadFile, File, HTTPException
from database.database import CompanyData as CompanyDataModel, VectorDB, async_session, get_session
from database.schemas import CompanyData
from sqlalchemy import select
from typing import List, Dict, Any
import io
import openai
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload Excel or CSV file containing company or product data.
    
    Company Data: File must contain a column named "公司名稱" (Company Name).
    Product Data: File must contain an industry category column (e.g., '產業別', 'industry_category').
    """
    file_id = str(uuid.uuid4())

    # Check file extension
    filename = (file.filename or "").lower()
    file_extension = filename.split('.')[-1] if '.' in filename else ''

    if file_extension not in ['csv', 'xlsx', 'xls']:
        raise HTTPException(status_code=400,
                            detail="Only CSV and Excel files are supported")

    try:
        # Read file content
        file_content = await file.read()
        file_io = io.BytesIO(file_content)

        # Read data based on file type
        if file_extension == 'csv':
            data = pd.read_csv(file_io)
        else:  # Excel files
            data = pd.read_excel(file_io)

        # Check if this is company data or product data
        if "公司名稱" in data.columns:
            # Process as company data
            companies_processed = await process_company_data(data, file_id)
            return {
                "message":
                f"Successfully processed {companies_processed} company records",
                "file_id": file_id,
                "records_processed": companies_processed,
                "data_type": "company"
            }
        else:
            # Process as product data
            products_processed = await process_product_data(data, file_id)
            return {
                "message":
                f"Successfully processed {products_processed} industry groups with embeddings",
                "file_id": file_id,
                "groups_processed": products_processed,
                "data_type": "product"
            }

    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Error processing file: {str(e)}")


async def process_company_data(data: pd.DataFrame, file_id: str) -> int:
    """
    Process company data from DataFrame and save to database.
    Returns the number of companies processed.
    """
    companies_processed = 0

    async for session in get_session():
        try:
            for index, row in data.iterrows():
                # Parse supplier evaluation information from string to dict
                supplier_eval_str = str(row.get("供應商評鑑資料", ""))
                supplier_eval_dict = {}
                if supplier_eval_str and supplier_eval_str != "nan":
                    try:
                        # Parse format like "品質:90, 交期:85, 合規:95, 服務:88"
                        for item in supplier_eval_str.split(", "):
                            if ":" in item:
                                key, value = item.split(":", 1)
                                supplier_eval_dict[key.strip()] = value.strip()
                    except:
                        supplier_eval_dict = {"raw_data": supplier_eval_str}

                # Parse keywords from string to list
                keywords_str = str(row.get("預設關鍵字", ""))
                keywords_list = []
                if keywords_str and keywords_str != "nan":
                    # Split by semicolon and clean up
                    keywords_list = [
                        kw.strip() for kw in keywords_str.split(";")
                        if kw.strip()
                    ]

                # Create company data object
                company_data = CompanyData(
                    CompanyName_CH=str(row.get("公司名稱", "")),
                    CompanyName_EN=str(row.get("公司英文名稱", "")),
                    Enterprise_Number=str(row.get("企業編號", "")),
                    Industry_category=str(row.get("產業類別", "")),
                    Supplier_evaluation_information=supplier_eval_dict,
                    Phhone_Number=str(row.get("電話", "")),
                    Address=str(row.get("地址", "")),
                    Website=str(row.get("公司網站", "")),
                    Email=str(row.get("E-mail", "")),
                    keywords=keywords_list,
                    Score=1.0)  # Default score since not in the Excel columns

                # Create database model instance
                db_company = CompanyDataModel(
                    id=str(uuid.uuid4()),
                    CompanyName_CH=company_data.CompanyName_CH,
                    CompanyName_EN=company_data.CompanyName_EN,
                    Enterprise_Number=company_data.Enterprise_Number,
                    Industry_category=company_data.Industry_category,
                    Supplier_evaluation_information=company_data.
                    Supplier_evaluation_information,
                    Phhone_Number=company_data.Phhone_Number,
                    Address=company_data.Address,
                    Website=company_data.Website,
                    Email=company_data.Email,
                    Keywords=company_data.keywords,
                    Score=company_data.Score)

                # Add to session
                session.add(db_company)
                companies_processed += 1

            # Commit all changes
            await session.commit()

        except Exception as e:
            await session.rollback()
            raise e

    return companies_processed


async def process_product_data(data: pd.DataFrame, file_id: str) -> int:
    """
    Process product data from DataFrame, classify by industry category (產業別), and create embeddings.
    Returns the number of industry groups processed.
    """
    # Group data by industry category (產業別)
    if '產業別' not in data.columns:
        # If no industry category column, try to find similar columns
        industry_columns = [
            col for col in data.columns
            if 'industry' in col.lower() or '產業' in col or '類別' in col
        ]
        if not industry_columns:
            raise HTTPException(
                status_code=400,
                detail=
                "Product data must contain an industry category column (e.g., '產業別', 'industry_category')"
            )
        industry_col = industry_columns[0]
    else:
        industry_col = '產業別'

    # Group data by industry category
    industry_groups = data.groupby(industry_col)
    groups_processed = 0

    async for session in get_session():
        try:
            for industry_category, group_data in industry_groups:
                # Create text content for embedding
                text_content = create_product_text_content(
                    industry_category, group_data)

                # Generate embedding
                embedding = await generate_embedding(text_content)

                # Create metadata
                metadata = {
                    "industry_category": str(industry_category),
                    "file_id": file_id,
                    "record_count": len(group_data),
                    "columns": list(group_data.columns),
                    "data_sample": group_data.head(3).to_dict(
                        'records')  # First 3 records as sample
                }

                # Create VectorDB entry
                vector_entry = VectorDB(id=str(uuid.uuid4()),
                                        filter={
                                            "industry_category":
                                            str(industry_category),
                                            "file_id":
                                            file_id
                                        },
                                        embedding=embedding,
                                        metadata_json=metadata)

                session.add(vector_entry)
                groups_processed += 1

            # Commit all changes
            await session.commit()

        except Exception as e:
            await session.rollback()
            raise e

    return groups_processed


def create_product_text_content(industry_category: str,
                                group_data: pd.DataFrame) -> str:
    """
    Create text content from industry group data for embedding generation.
    """
    text_parts = [f"Industry Category: {industry_category}"]

    # Add all text data from the group
    for column in group_data.columns:
        if group_data[column].dtype == 'object':  # Text columns
            unique_values = group_data[column].dropna().unique()
            if len(unique_values) > 0:
                text_parts.append(
                    f"{column}: {', '.join(map(str, unique_values))}")

    return " | ".join(text_parts)


async def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding for the given text using OpenAI API.
    """
    try:
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Generate embedding
        response = client.embeddings.create(model="text-embedding-3-small",
                                            input=text)

        return response.data[0].embedding

    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Failed to generate embedding: {str(e)}")
