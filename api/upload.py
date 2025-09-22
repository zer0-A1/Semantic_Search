import pandas as pd
import uuid
import json
from fastapi import APIRouter, UploadFile, File, HTTPException
from database.database import VectorDB, async_session, get_session
from sqlalchemy import select
from typing import List, Dict, Any
import io
import openai
import os
from dotenv import load_dotenv
from datetime import date
import numpy as np

load_dotenv()

router = APIRouter()


def calculate_product_score(row: pd.Series, industry_category: str) -> float:
    """
    Calculate score for product data.
    Start with 1.0, minus 0.05 for each empty column.
    Minus 0.1 if expire date is before today or issue date is after today.
    """
    score = 1.0
    empty_penalty = 0.05
    date_penalty = 0.1

    # Check each column for empty values
    for column in row.index:
        value = row[column]
        if pd.isna(value) or value == "" or str(value).strip() == "":
            score -= empty_penalty

    # Check date validation
    today = date.today()

    # Check expire date
    if 'expire_date' in row.index and not pd.isna(row['expire_date']):
        try:
            expire_date = pd.to_datetime(row['expire_date']).date()
            if expire_date < today:
                score -= date_penalty
        except:
            pass  # If date parsing fails, skip penalty

    # Check issue date
    if 'issue_date' in row.index and not pd.isna(row['issue_date']):
        try:
            issue_date = pd.to_datetime(row['issue_date']).date()
            if issue_date > today:
                score -= date_penalty
        except:
            pass  # If date parsing fails, skip penalty

    return max(0.0, score)  # Ensure score doesn't go below 0


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload Excel or CSV file containing company or product data.
    Automatically updates VectorDB for product data.
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


async def process_product_data(data: pd.DataFrame, file_id: str) -> int:
    """
    Process product data from DataFrame, classify by industry category (產業別), and create embeddings.
    Updates existing VectorDB entries or creates new ones as needed.
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

    # Process each group and update VectorDB
    for industry_category, group_data in industry_groups:
        # Calculate scores and collect per-product metrics for each record in the group
        scores = []

        # Try to locate common columns for products, company, quantity and tags once per group
        product_cols_detect = [
            col for col in group_data.columns
            if any(k in str(col) for k in
                   ["問卷編號", "產品編號", "產品代號", "product_id", "product", "sku"])
        ]
        company_cols_detect = [
            col for col in group_data.columns
            if any(k in str(col)
                   for k in ["客戶名稱", "公司名稱", "company", "company_name"])
        ]
        quantity_cols_detect = [
            col for col in group_data.columns
            if any(k in str(col).lower() for k in ["數量", "quantity", "qty"])
        ]
        tags_cols_detect = [
            col for col in group_data.columns
            if any(k in str(col).lower() for k in ["tags", "tag", "標籤"])
        ]

        product_col_detect = product_cols_detect[
            0] if product_cols_detect else None
        company_col_detect = company_cols_detect[
            0] if company_cols_detect else None
        quantity_col_detect = quantity_cols_detect[
            0] if quantity_cols_detect else None
        tags_col_detect = tags_cols_detect[0] if tags_cols_detect else None

        # Identify numeric columns (excluding obvious id/company columns)
        id_like = {
            c
            for c in group_data.columns if any(
                k in str(c).lower() for k in ["id", "編號", "代號", "問卷", "sku"])
        }
        company_like = {
            c
            for c in group_data.columns
            if any(k in str(c).lower() for k in ["company", "公司", "客戶"])
        }
        ignore_cols = id_like.union(company_like)

        numeric_fields = set()
        for col in group_data.columns:
            if col in ignore_cols:
                continue
            non_na = group_data[col].dropna().head(20)
            if non_na.empty:
                continue
            parsable = 0
            for v in non_na:
                try:
                    _ = float(v)
                    parsable += 1
                except Exception:
                    pass
            if parsable >= max(3, int(len(non_na) * 0.6)):
                numeric_fields.add(str(col))

        product_metrics = []

        for index, row in group_data.iterrows():
            score = calculate_product_score(row, industry_category)
            scores.append(score)

            pid_val = row.get(
                product_col_detect) if product_col_detect else None
            pid = None if pd.isna(pid_val) else str(pid_val).strip()

            cname_val = row.get(
                company_col_detect) if company_col_detect else None
            cname = None if pd.isna(cname_val) else str(cname_val).strip()

            quantity_val = None
            if quantity_col_detect:
                qv = row.get(quantity_col_detect)
                try:
                    if not pd.isna(qv):
                        quantity_val = float(qv)
                except Exception:
                    quantity_val = None

            tags_val = None
            if tags_col_detect:
                tv = row.get(tags_col_detect)
                if not pd.isna(tv):
                    tags_val = [
                        t.strip() for t in str(tv).replace('|', ',').split(',')
                        if t.strip()
                    ]

            if pid:
                # Include all fields for flexible metric queries
                fields = {}
                for col in group_data.columns:
                    val = row.get(col)
                    if pd.isna(val):
                        continue
                    key = str(col).strip()
                    try:
                        if key in numeric_fields:
                            fields[key.lower()] = float(val)
                        else:
                            fields[key.lower()] = str(val).strip()
                    except Exception:
                        fields[key.lower()] = str(val)

                product_metrics.append({
                    "product_id": pid,
                    "company": cname,
                    "quantity": quantity_val,
                    "quality_score": float(score),
                    "tags": tags_val or [],
                    "fields": fields
                })

        # Calculate average score for the group
        average_score = np.mean(scores) if scores else 1.0

        # Create text content for embedding (include average score)
        text_content = create_product_text_content(industry_category,
                                                   group_data, average_score)

        # Generate embedding
        embedding = await generate_embedding(text_content)

        # Create metadata
        metadata = {
            "industry_category": str(industry_category),
            "file_id": file_id,
            "record_count": len(group_data),
            "average_score": float(average_score),
            "individual_scores": [float(s) for s in scores],
            "columns": list(group_data.columns),
            "data_sample":
            group_data.head(3).to_dict('records'),  # First 3 records as sample
            "last_updated": pd.Timestamp.now().isoformat()
        }

        # Attach per-product metrics if available
        if product_metrics:
            metadata["product_metrics"] = product_metrics
            if numeric_fields:
                metadata["numeric_fields"] = [nf for nf in numeric_fields]

        # Derive product identifiers and mapping to company names for better search resolution
        try:
            product_cols = [
                col for col in group_data.columns if any(
                    k in str(col) for k in
                    ["問卷編號", "產品編號", "產品代號", "product_id", "product", "sku"])
            ]
            company_cols = [
                col for col in group_data.columns
                if any(k in str(col)
                       for k in ["客戶名稱", "公司名稱", "company", "company_name"])
            ]

            product_col = product_cols[0] if product_cols else None
            company_col = company_cols[0] if company_cols else None

            product_ids: List[str] = []
            product_to_company: Dict[str, str] = {}

            if product_col:
                for _, row in group_data.iterrows():
                    pid_val = row.get(product_col)
                    if pd.isna(pid_val):
                        continue
                    pid = str(pid_val).strip()
                    if not pid:
                        continue
                    product_ids.append(pid)
                    if company_col:
                        cname_val = row.get(company_col)
                        if not pd.isna(cname_val):
                            product_to_company[pid] = str(cname_val).strip()

            if product_ids:
                # Keep unique order
                seen = set()
                unique_products = []
                for p in product_ids:
                    if p not in seen:
                        unique_products.append(p)
                        seen.add(p)

                metadata["product_ids"] = unique_products
                if product_to_company:
                    metadata["product_to_company"] = product_to_company
        except Exception:
            # Non-fatal enrichment; ignore if any error occurs
            pass

        # Update or create VectorDB entry
        await update_or_create_vector_entry(industry_category, embedding,
                                            metadata)
        groups_processed += 1

    return groups_processed


async def update_or_create_vector_entry(industry_category: str,
                                        embedding: List[float],
                                        metadata: dict):
    """
    Update existing VectorDB entry for the industry category or create a new one.
    This ensures VectorDB is always up-to-date with the latest data.
    """
    async for session in get_session():
        try:
            # Check if entry exists for this industry category
            stmt = select(VectorDB).where(VectorDB.filter == industry_category)
            result = await session.execute(stmt)
            existing_entry = result.scalar_one_or_none()

            if existing_entry:
                # Update existing entry
                existing_entry.embedding = embedding
                existing_entry.metadata_json = metadata
                print(
                    f"Updated VectorDB entry for industry: {industry_category}"
                )
            else:
                # Create new entry
                vector_entry = VectorDB(id=str(uuid.uuid4()),
                                        filter=industry_category,
                                        embedding=embedding,
                                        metadata_json=metadata)
                session.add(vector_entry)
                print(
                    f"Created new VectorDB entry for industry: {industry_category}"
                )

            await session.commit()

        except Exception as e:
            await session.rollback()
            raise e


def create_product_text_content(industry_category: str,
                                group_data: pd.DataFrame,
                                average_score: float) -> str:
    """
    Create text content from industry group data for embedding generation.
    """
    text_parts = [
        f"Industry Category: {industry_category}",
        f"Average Score: {average_score:.2f}"
    ]

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


@router.post("/vectordb/refresh")
async def refresh_vectordb():
    """
    Manually refresh all VectorDB entries by regenerating embeddings.
    This can be useful if you want to update all existing entries with new embeddings.
    """
    try:
        async for session in get_session():
            # Get all existing VectorDB entries
            stmt = select(VectorDB)
            result = await session.execute(stmt)
            vector_entries = result.scalars().all()

            updated_count = 0
            for entry in vector_entries:
                try:
                    # Regenerate embedding for existing text content
                    metadata = entry.metadata_json or {}
                    industry_category = entry.filter

                    # Recreate text content from metadata
                    text_content = f"Industry Category: {industry_category}"
                    if 'average_score' in metadata:
                        text_content += f" | Average Score: {metadata['average_score']:.2f}"
                    if 'data_sample' in metadata and metadata['data_sample']:
                        # Add sample data to text content
                        sample_texts = []
                        for sample in metadata[
                                'data_sample'][:3]:  # First 3 samples
                            for key, value in sample.items():
                                if isinstance(value, str) and value.strip():
                                    sample_texts.append(f"{key}: {value}")
                        if sample_texts:
                            text_content += " | " + " | ".join(sample_texts)

                    # Generate new embedding
                    new_embedding = await generate_embedding(text_content)

                    # Update entry
                    entry.embedding = new_embedding
                    metadata['last_refreshed'] = pd.Timestamp.now().isoformat()
                    entry.metadata_json = metadata

                    updated_count += 1

                except Exception as e:
                    print(f"Failed to refresh entry {entry.id}: {e}")
                    continue

            await session.commit()

        return {
            "message":
            f"Successfully refreshed {updated_count} VectorDB entries",
            "entries_updated": updated_count
        }

    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Failed to refresh VectorDB: {str(e)}")


@router.get("/vectordb/stats")
async def get_vectordb_stats():
    """
    Get statistics about the current VectorDB entries.
    """
    try:
        async for session in get_session():
            # Get all VectorDB entries
            stmt = select(VectorDB)
            result = await session.execute(stmt)
            vector_entries = result.scalars().all()

            # Calculate statistics
            total_entries = len(vector_entries)
            industry_categories = set(entry.filter for entry in vector_entries)

            # Get metadata statistics
            total_records = 0
            avg_scores = []

            for entry in vector_entries:
                metadata = entry.metadata_json or {}
                if 'record_count' in metadata:
                    total_records += metadata['record_count']
                if 'average_score' in metadata:
                    avg_scores.append(metadata['average_score'])

            avg_score = np.mean(avg_scores) if avg_scores else 0

            return {
                "total_vector_entries":
                total_entries,
                "unique_industry_categories":
                len(industry_categories),
                "industry_categories":
                list(industry_categories),
                "total_records_represented":
                total_records,
                "average_quality_score":
                round(avg_score, 3),
                "last_updated_entries": [
                    {
                        "industry":
                        entry.filter,
                        "last_updated":
                        entry.metadata_json.get('last_updated', 'Unknown')
                        if entry.metadata_json else 'Unknown'
                    } for entry in vector_entries[-5:]  # Last 5 entries
                ]
            }

    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Failed to get VectorDB stats: {str(e)}")


@router.delete("/vectordb/industry/{industry_category}")
async def delete_vectordb_entry(industry_category: str):
    """
    Delete VectorDB entry for a specific industry category.
    """
    try:
        async for session in get_session():
            # Find and delete the entry
            stmt = select(VectorDB).where(VectorDB.filter == industry_category)
            result = await session.execute(stmt)
            entry = result.scalar_one_or_none()

            if not entry:
                raise HTTPException(
                    status_code=404,
                    detail=
                    f"No VectorDB entry found for industry: {industry_category}"
                )

            await session.delete(entry)
            await session.commit()

            return {
                "message":
                f"Successfully deleted VectorDB entry for industry: {industry_category}",
                "deleted_industry": industry_category
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete VectorDB entry: {str(e)}")
