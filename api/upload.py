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
from datetime import date, datetime
import numpy as np
import math

load_dotenv()

router = APIRouter()


def convert_to_date_style(date_str):
    # If already a date/datetime, normalize to date
    try:
        if isinstance(date_str, datetime):
            return date_str.date()
        if isinstance(date_str, date):
            return date_str
    except Exception:
        pass

    # Handle None, empty, or invalid values
    if pd.isna(date_str) or str(date_str).strip() == '' or str(
            date_str).lower() in ['nan', 'none', 'null']:
        raise ValueError(f"Invalid date value: {date_str}")

    # Handle Excel serial dates (numeric values)
    try:
        if isinstance(date_str, (int, float)) and not pd.isna(date_str):
            # Excel serial date: days since 1900-01-01 (with leap year bug)
            # Excel incorrectly treats 1900 as a leap year
            excel_epoch = datetime(1899, 12, 30)  # Excel's epoch
            if 1 <= date_str <= 2958465:  # Reasonable date range
                result_date = excel_epoch + pd.Timedelta(days=date_str)
                if result_date.year > 1900:  # Valid date after 1900
                    return result_date.date()
    except Exception:
        pass

    # Try pandas to_datetime with different dayfirst settings
    try:
        # First try with dayfirst=False (US format: MM/DD/YYYY)
        ts = pd.to_datetime(str(date_str), errors='coerce', dayfirst=False)
        if pd.notna(ts) and ts.year > 1900:  # Valid date after 1900
            return ts.date()
    except Exception:
        pass

    try:
        # Then try with dayfirst=True (EU format: DD/MM/YYYY)
        ts = pd.to_datetime(str(date_str), errors='coerce', dayfirst=True)
        if pd.notna(ts) and ts.year > 1900:  # Valid date after 1900
            return ts.date()
    except Exception:
        pass

    # Try explicit formats as fallback
    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m/%d/%Y",  # US format
        "%d/%m/%Y",  # EU format
        "%d-%m-%Y",
        "%m-%d-%Y",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(str(date_str), fmt)
            if dt.year > 1900:  # Valid date after 1900
                return dt.date()
        except Exception:
            continue

    raise ValueError(f"Invalid date format: {date_str}")


def calculate_product_score(row: pd.Series) -> float:
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
            expire_date = convert_to_date_style(row['expire_date'])
            print(f"Debug - Expire date: {expire_date}, Today: {today}")
            if expire_date < today:
                score -= date_penalty
                print(f"Debug - Expire date penalty applied: {score}")
        except Exception as e:
            print(
                f"Debug - Expire date parsing failed: {row['expire_date']}, Error: {e}"
            )
            # If date parsing fails, skip penalty

    # Check issue date
    if 'issue_date' in row.index and not pd.isna(row['issue_date']):
        try:
            issue_date = convert_to_date_style(row['issue_date'])
            print(f"Debug - Issue date: {issue_date}, Today: {today}")
            if issue_date > today:
                score -= date_penalty
                print(f"Debug - Issue date penalty applied: {score}")
        except Exception as e:
            print(
                f"Debug - Issue date parsing failed: {row['issue_date']}, Error: {e}"
            )
            # If date parsing fails, skip penalty
    print(f"Debug - Score: {score}")
    return max(0.0, score)  # Ensure score doesn't go below 0


def _extract_country_from_row(row: pd.Series) -> str | None:
    """
    Heuristically extract a country string from common columns like country/address.
    Returns a raw country string if found; otherwise None.
    """
    candidate_cols = []
    for col in row.index:
        col_l = str(col).lower()
        if any(k in col_l for k in [
                "country", "國家", "國別", "國籍", "地址", "address", "addr", "所在地",
                "公司地址", "營業地址"
        ]):
            candidate_cols.append(col)
    for col in candidate_cols:
        try:
            val = row.get(col)
            if pd.isna(val):
                continue
            s = str(val).strip()
            if s:
                return s
        except Exception:
            continue
    return None


def _convert_excel_date_to_string(date_value):
    """
    Convert Excel serial date or other date formats to YYYY-MM-DD string.
    Returns the original value if conversion fails.
    """
    try:
        # If already a date/datetime, format to string
        if isinstance(date_value, datetime):
            return date_value.strftime('%Y-%m-%d')
        if isinstance(date_value, date):
            return date_value.strftime('%Y-%m-%d')

        # Handle Excel serial dates (numeric values)
        if isinstance(date_value, (int, float)) and not pd.isna(date_value):
            # Excel serial date: days since 1900-01-01 (with leap year bug)
            excel_epoch = datetime(1899, 12, 30)  # Excel's epoch
            if 1 <= date_value <= 2958465:  # Reasonable date range
                result_date = excel_epoch + pd.Timedelta(days=date_value)
                if result_date.year > 1900:  # Valid date after 1900
                    return result_date.strftime('%Y-%m-%d')

        # Try pandas to_datetime for string dates
        ts = pd.to_datetime(str(date_value), errors='coerce')
        if pd.notna(ts) and ts.year > 1900:
            return ts.strftime('%Y-%m-%d')

    except Exception:
        pass

    # Return original value if conversion fails
    return date_value


def _sanitize_data_sample_for_json(data_sample):
    """
    Convert Excel serial dates in data_sample to readable string format.
    """
    if not data_sample:
        return data_sample

    sanitized = []
    for record in data_sample:
        sanitized_record = {}
        for key, value in record.items():
            # Check if this looks like a date column
            if any(date_key in str(key).lower()
                   for date_key in ['date', '日期', 'expire', 'issue']):
                sanitized_record[key] = _convert_excel_date_to_string(value)
            else:
                sanitized_record[key] = value
        sanitized.append(sanitized_record)

    return sanitized


def _sanitize_json(value):
    """
    Recursively sanitize Python objects to be JSON-safe for Postgres JSON:
    - Replace NaN/Inf/-Inf with None
    - Convert pandas/NumPy scalar types to native Python types
    - Convert non-serializable values to strings as a last resort
    """

    if value is None:
        return None
    # Basic numeric types
    if isinstance(value, (int, )):
        return int(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else float(value)
    # NumPy scalar types
    if isinstance(value, (np.integer, )):
        return int(value)
    if isinstance(value, (np.floating, )):
        f = float(value)
        return None if not math.isfinite(f) else f
    # Strings
    if isinstance(value, str):
        return value
    # Dict
    if isinstance(value, dict):
        return {str(k): _sanitize_json(v) for k, v in value.items()}
    # List/Tuple/Set
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_json(v) for v in list(value)]
    # Pandas Timestamp/NaT and others -> string or None
    try:
        # Handle pandas.NaT and np.nan
        if pd.isna(value):
            return None
    except Exception:
        pass
    # Fallback to string
    try:
        return str(value)
    except Exception:
        return None


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

        products_processed, filters_processed = await process_product_data(
            data, file_id)
        return {
            "message":
            f"Successfully processed {products_processed} industry groups with embeddings",
            "file_id": file_id,
            "groups_processed": products_processed,
            "filters": filters_processed,
            "data_type": "product"
        }

    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Error processing file: {str(e)}")


async def process_product_data(data: pd.DataFrame,
                               file_id: str) -> tuple[int, List[str]]:
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
    filters_processed: List[str] = []

    # Process each group and update VectorDB
    for industry_category, group_data in industry_groups:
        # Track filter value processed for response
        try:
            filters_processed.append(str(industry_category))
        except Exception:
            pass
        # Precompute per-row normalized country for subgrouping
        def _normalize_country(v: str | None) -> str | None:
            if v is None:
                return None
            try:
                s = str(v)
            except Exception:
                return None
            s = s.replace('\n', ' ').strip()
            s = ' '.join(s.split())
            return s if s else None

        # Convert all date columns in the group data to readable format
        def convert_dataframe_dates(df):
            """Convert all date columns in DataFrame to readable string format."""
            df_copy = df.copy()
            for col in df_copy.columns:
                # Check if this looks like a date column
                if any(date_key in str(col).lower()
                       for date_key in ['date', '日期', 'expire', 'issue']):
                    df_copy[col] = df_copy[col].apply(
                        _convert_excel_date_to_string)
            return df_copy

        # Apply date conversion to the entire group data
        group_data = convert_dataframe_dates(group_data)

        # Calculate scores and collect per-product metrics for each record in the group (full group first)
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

        product_col_detect = product_cols_detect[
            0] if product_cols_detect else None
        company_col_detect = company_cols_detect[
            0] if company_cols_detect else None
        quantity_col_detect = quantity_cols_detect[
            0] if quantity_cols_detect else None

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
        countries_in_group = set()
        row_countries: Dict[int, str | None] = {}

        for index, row in group_data.iterrows():
            score = calculate_product_score(row)
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

            # Extract country from row
            country_val = _extract_country_from_row(row)
            if country_val:
                try:
                    cn = _normalize_country(country_val)
                    countries_in_group.add(cn)
                    row_countries[index] = cn
                except Exception:
                    row_countries[index] = None
            else:
                row_countries[index] = None

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
                # Derive quality score fallback if no tags provided
                quality_score_value = float(score)
                if fields:
                    # Try to compute average of delivery rate and return rate (percentages)
                    def _parse_rate(v):
                        try:
                            f = float(v)
                            # Heuristic: >1 means percentage 0-100
                            return f / 100.0 if f > 1.0 else f
                        except Exception:
                            return None

                    # Common field name variants
                    delivery_keys = [
                        "delivery_rate", "delivery rate", "準時交貨率", "交貨率",
                        "on_time_delivery_rate", "otd", "delivery%"
                    ]
                    return_keys = [
                        "return_rate", "return rate", "退貨率", "退貨%", "rma_rate"
                    ]
                    d_vals = []
                    r_vals = []
                    lowered = {str(k).lower(): v for k, v in fields.items()}
                    for k in list(lowered.keys()):
                        if any(
                                k.startswith(dk) or dk in k
                                for dk in delivery_keys):
                            parsed = _parse_rate(lowered[k])
                            if parsed is not None:
                                d_vals.append(parsed)
                        if any(
                                k.startswith(rk) or rk in k
                                for rk in return_keys):
                            parsed = _parse_rate(lowered[k])
                            if parsed is not None:
                                r_vals.append(parsed)
                    d = d_vals[0] if d_vals else None
                    r = r_vals[0] if r_vals else None
                    if d is not None and r is not None:
                        try:
                            quality_score_value = float(
                                max(0.0, min(1.0, (d + r) / 2.0)))
                        except Exception:
                            pass
                    elif d is not None:
                        quality_score_value = float(max(0.0, min(1.0, d)))
                    elif r is not None:
                        quality_score_value = float(max(0.0, min(1.0, r)))

                product_metrics.append({
                    "product_id":
                    pid,
                    "company":
                    cname,
                    "quantity":
                    quantity_val,
                    "quality_score":
                    float(quality_score_value),
                    "score":
                    float(score),
                    "country":
                    country_val,
                    "fields":
                    fields
                })

        # If countries present, create subgroup entries per country; otherwise one entry
        subgroup_keys = sorted({c for c in countries_in_group if c}) or [None]

        for cn in subgroup_keys:
            if cn is None:
                mask = [row_countries.get(i) is None for i in group_data.index]
            else:
                mask = [row_countries.get(i) == cn for i in group_data.index]
            # Build subgroup DataFrame
            try:
                mask_series = pd.Series(mask, index=group_data.index)
                subgroup = group_data[mask_series]
            except Exception:
                subgroup = group_data

            # Apply date conversion to subgroup as well
            subgroup = convert_dataframe_dates(subgroup)

            # Recompute scores for subgroup
            scores_sub = []
            for _, r in subgroup.iterrows():
                scores_sub.append(calculate_product_score(r))
            average_score = np.mean(scores_sub) if scores_sub else 1.0

            # Create text content for embedding (include average score)
            text_content = create_product_text_content(industry_category,
                                                       subgroup, average_score)
            embedding = await generate_embedding(text_content)

            # Subset product_metrics to this country
            if cn is None:
                pm_sub = [m for m in product_metrics if not m.get('country')]
            else:
                pm_sub = [
                    m for m in product_metrics
                    if m.get('country') and str(m.get('country')).strip() == cn
                ]

            # Create metadata for subgroup
            # Dates are already converted, so we can use to_dict directly
            metadata = {
                "industry_category": str(industry_category),
                "group_country": cn,
                "file_id": file_id,
                "record_count": int(len(subgroup)),
                "average_score": float(average_score),
                "individual_scores": [float(s) for s in scores_sub],
                "columns": list(subgroup.columns),
                "data_sample": subgroup.head(3).to_dict('records'),
                "last_updated": pd.Timestamp.now().isoformat(),
                "countries": [cn] if cn else []
            }

            if pm_sub:
                metadata["product_metrics"] = pm_sub
                if numeric_fields:
                    metadata["numeric_fields"] = [nf for nf in numeric_fields]

            # Derive product ids mapping within subgroup
            try:
                product_cols = [
                    col for col in subgroup.columns
                    if any(k in str(col) for k in [
                        "問卷編號", "產品編號", "產品代號", "product_id", "product", "sku"
                    ])
                ]
                company_cols = [
                    col for col in subgroup.columns if any(
                        k in str(col)
                        for k in ["客戶名稱", "公司名稱", "company", "company_name"])
                ]
                product_col = product_cols[0] if product_cols else None
                company_col = company_cols[0] if company_cols else None
                product_ids: List[str] = []
                product_to_company: Dict[str, str] = {}
                if product_col:
                    for _, row in subgroup.iterrows():
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
                                product_to_company[pid] = str(
                                    cname_val).strip()
                if product_ids:
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
                pass

            # Build filter label including country for better discoverability
            entry_filter = f"{industry_category}|{cn}" if cn else str(
                industry_category)
            await update_or_create_vector_entry(entry_filter, embedding,
                                                metadata)
            groups_processed += 1

    return groups_processed, filters_processed


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

            # sanitize metadata before persistence
            safe_metadata = _sanitize_json(metadata)

            if existing_entry:
                # Update existing entry
                existing_entry.embedding = embedding
                existing_entry.metadata_json = safe_metadata
                print(
                    f"Updated VectorDB entry for industry: {industry_category}"
                )
            else:
                # Create new entry
                vector_entry = VectorDB(id=str(uuid.uuid4()),
                                        filter=industry_category,
                                        embedding=embedding,
                                        metadata_json=safe_metadata)
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
                    entry.metadata_json = _sanitize_json(metadata)

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
