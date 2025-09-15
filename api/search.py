import numpy as np
from fastapi import APIRouter, HTTPException, Query
from database.database import VectorDB, get_session, clear_statement_cache, save_search_query_and_results
from database.schemas import SearchRequest, SearchResult
from sqlalchemy import select, or_
from typing import List, Dict, Any
import openai
import os
from dotenv import load_dotenv
import re

load_dotenv()

router = APIRouter()


def parse_filters_string(filters_str: str) -> Dict[str, List[str]]:
    """
    Parse filters string into dictionary format.
    Simple format: "扣件" -> {"industry_category": ["扣件"]}
    Complex format: "industry:FOOD,FOOD2;country:VN,TH" -> {"industry": ["FOOD", "FOOD2"], "country": ["VN", "TH"]}
    """
    filters = {}

    if not filters_str or filters_str.strip() == "":
        return filters

    try:
        # Check if it's a complex format (contains colons)
        if ':' in filters_str:
            # Complex format: "industry:FOOD,FOOD2;country:VN,TH"
            filter_pairs = filters_str.split(';')
            for pair in filter_pairs:
                if ':' in pair:
                    key, values = pair.split(':', 1)
                    key = key.strip()
                    values = [
                        v.strip() for v in values.split(',') if v.strip()
                    ]
                    filters[key] = values
        else:
            # Simple format: "扣件" -> treat as industry category
            filters["industry_category"] = [filters_str.strip()]
    except Exception:
        # If parsing fails, return empty filters
        pass

    return filters


@router.post("/search")
async def semantic_search(request: SearchRequest):
    """
    Perform semantic search with fuzzy matching and multi-factor sorting.
    
    Supports:
    - Natural language queries
    - Semantic similarity search
    - Numeric tolerance matching
    - Multi-factor sorting (Completeness + SemanticSim + NumericFit)
    """
    try:
        # Extract parameters from request
        query = request.query_text
        filters_str = request.filters
        top_k = request.top_k

        # Parse filters string to dictionary
        filters = parse_filters_string(filters_str)

        # Step 1: Filter VectorDB by filter values first
        filtered_groups = await filter_vectordb_by_filters(filters)

        if not filtered_groups:
            return {"results": []}

        # Step 2: Generate embedding for the query
        query_embedding = await generate_embedding(query)

        # Step 3: Perform semantic search within filtered groups
        vector_results = await semantic_search_within_groups(
            query, query_embedding, filtered_groups, top_k)

        # Step 4: Convert to response format
        formatted_results = []
        vector_ids = []

        for result in vector_results[:top_k]:
            # Debug: Print result structure
            print(f"Debug - Result keys: {list(result.keys())}")
            print(
                f"Debug - Company name value: {result.get('company_name')} (type: {type(result.get('company_name'))})"
            )

            # Ensure company name is never None or empty
            company_name = result.get('company_name') or 'Unknown Company'
            if not isinstance(company_name, str):
                company_name = str(
                    company_name
                ) if company_name is not None else 'Unknown Company'

            formatted_result = SearchResult(
                company=company_name,
                product=result.get('product_name'),
                completeness_score=int(result['completeness_score'] * 100),
                semantic_score=round(result['semantic_similarity'], 2),
                # numeric_gap=calculate_numeric_gap(query, result),
                doc_status=result['document_status'],
                total_score=int(result['overall_score'] * 100))
            formatted_results.append(formatted_result)

            # Collect vector IDs for database storage
            vector_ids.append(result['metadata']['vector_id'])

        # Save query and results to database
        try:
            query_id = await save_search_query_and_results(
                query_text=query,
                filters=filters_str,
                top_k=top_k,
                results=formatted_results,
                vector_ids=vector_ids)
            print(f"Search query saved with ID: {query_id}")
        except Exception as e:
            print(f"Failed to save search query: {e}")
            # Continue even if saving fails

        # Return results in the specified format
        return {"results": formatted_results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


async def filter_vectordb_by_filters(
        filters: Dict[str, List[str]]) -> List[VectorDB]:
    """
    Step 1: Filter VectorDB by filter values to find correct groups.
    Returns list of VectorDB entries that match the filter criteria.
    """
    filtered_entries = []
    max_retries = 3

    for attempt in range(max_retries):
        try:
            async for session in get_session():
                try:
                    # Build base query
                    stmt = select(VectorDB)

                    # Apply industry filter (both "industry" and "industry_category" keys)
                    industry_filters = filters.get(
                        "industry", []) + filters.get("industry_category", [])
                    if industry_filters:
                        industry_conditions = []
                        for industry in industry_filters:
                            # Use string matching since filter is String type
                            industry_conditions.append(
                                VectorDB.filter.ilike(f"%{industry}%"))
                        stmt = stmt.where(or_(*industry_conditions))

                    # Apply country filter (if country info is stored in metadata)
                    if filters.get("country"):
                        country_conditions = []
                        for country in filters["country"]:
                            country_conditions.append(
                                VectorDB.metadata_json.op('->>')(
                                    'country').ilike(f"%{country}%"))
                        stmt = stmt.where(or_(*country_conditions))

                    # Execute query
                    result = await session.execute(stmt)
                    filtered_entries = result.scalars().all()
                    return filtered_entries  # Return immediately on success

                except Exception as e:
                    await session.rollback()
                    print(f"Database query error (attempt {attempt + 1}): {e}")

                    # If it's a cached statement error, clear cache and retry
                    if "InvalidCachedStatementError" in str(e):
                        await clear_statement_cache()
                        break  # Break out of session loop to retry
                    else:
                        raise e  # Re-raise if it's a different error

        except Exception as e:
            print(f"Session error (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                print("Max retries reached, returning empty list")
                return []
            # Wait a bit before retrying
            import asyncio
            await asyncio.sleep(0.1 * (attempt + 1))

    return filtered_entries


async def semantic_search_within_groups(query: str,
                                        query_embedding: List[float],
                                        filtered_groups: List[VectorDB],
                                        top_k: int) -> List[Dict[str, Any]]:
    """
    Step 2: Perform semantic search within the filtered groups.
    Returns ranked results based on semantic similarity.
    """
    results = []

    # Calculate similarity for each filtered group
    for vector_entry in filtered_groups:
        # Calculate cosine similarity
        similarity = calculate_cosine_similarity(query_embedding,
                                                 vector_entry.embedding)

        # Get metadata
        metadata = vector_entry.metadata_json or {}

        # Debug: Print metadata structure to understand the data
        print(
            f"Debug - Metadata keys: {list(metadata.keys()) if metadata else 'Empty metadata'}"
        )
        print(f"Debug - Filter value: {vector_entry.filter}")

        # Get completeness score from metadata (already calculated in upload.py)
        completeness = metadata.get('average_score', 0.5)

        # Calculate numeric fit
        # numeric_fit = calculate_numeric_fit_from_metadata(query, metadata)

        # Identify data gaps
        data_gaps = identify_data_gaps_from_metadata(metadata)

        # Determine document status
        doc_status = determine_document_status_from_metadata(metadata)

        # Calculate overall score
        overall_score = (completeness * 0.6 + similarity * 0.4)
        #  100 * numeric_fit * 0.0)

        # Extract company name from data_sample array
        data_sample = metadata.get('data_sample', [])
        company_name = 'Unknown Company'
        product_name = None

        if data_sample and len(data_sample) > 0:
            # Get the first record from data_sample
            first_record = data_sample[0]
            company_name = first_record.get('客戶名稱', 'Unknown Company')
            product_name = first_record.get('問卷編號')

        # Fallback to direct metadata fields if data_sample is empty
        if company_name == 'Unknown Company':
            company_name = (metadata.get('客戶名稱')
                            or metadata.get('company_name')
                            or metadata.get('company') or 'Unknown Company')

        if not product_name:
            product_name = (metadata.get('問卷編號')
                            or metadata.get('product_name')
                            or metadata.get('product')
                            or metadata.get('product_id'))

        results.append({
            "company_name": company_name,
            "product_name": product_name,
            "document_status": doc_status,
            "data_gaps": data_gaps,
            "completeness_score": completeness,
            "semantic_similarity": similarity,
            # "numeric_fit": numeric_fit,
            "overall_score": overall_score,
            "metadata": {
                "vector_id":
                vector_entry.id,
                "industry":
                (first_record.get('產業別')
                 if data_sample and len(data_sample) > 0 else
                 metadata.get('產業別') or metadata.get('industry_category')),
                "record_count":
                metadata.get('record_count', 0),
                "columns":
                metadata.get('columns', []),
                "data_sample":
                metadata.get('data_sample', []),
                "individual_scores":
                metadata.get('individual_scores', []),
                "filter":
                vector_entry.filter
            }
        })

    # Sort by overall score (descending)
    results.sort(key=lambda x: x['overall_score'], reverse=True)

    # Return only the top_k results
    return results[:top_k]


def calculate_cosine_similarity(embedding1: List[float],
                                embedding2: List[float]) -> float:
    """Calculate cosine similarity between two embeddings."""
    try:
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return float(similarity)

    except Exception:
        return 0.0


def calculate_numeric_fit_from_metadata(query: str,
                                        metadata: Dict[str, Any]) -> float:
    """Calculate numeric fit from vector metadata."""
    query_numbers = re.findall(r'\d+\.?\d*', query)
    if not query_numbers:
        return 0.5

    # Extract scores from data_sample
    data_sample = metadata.get('data_sample', [])
    scores = []

    if data_sample:
        for record in data_sample:
            # Collect all numeric scores from the record
            for key, value in record.items():
                if isinstance(
                        value,
                    (int, float)) and key not in ['quantity', 'category_code']:
                    scores.append(value)

    # Fallback to individual_scores if data_sample is empty
    if not scores:
        scores = metadata.get('individual_scores', [])

    if not scores:
        return 0.5

    best_fit = 0.0
    for query_num in query_numbers:
        query_val = float(query_num)
        for score in scores:
            diff = abs(query_val - score) / max(query_val, score, 1)
            fit = max(0, 1 - diff)
            best_fit = max(best_fit, fit)

    return best_fit


def identify_data_gaps_from_metadata(metadata: Dict[str, Any]) -> List[str]:
    """Identify data gaps from vector metadata."""
    gaps = []

    data_sample = metadata.get('data_sample', [])
    record_count = len(data_sample) if data_sample else metadata.get(
        'record_count', 0)

    if record_count == 0:
        gaps.append("No data records")

    # Check if we have enough data in data_sample
    if data_sample and len(data_sample) > 0:
        # Check if records have sufficient fields
        first_record = data_sample[0]
        required_fields = ['客戶名稱', '問卷編號', '產業別', '整體滿意度']
        missing_fields = [
            field for field in required_fields if field not in first_record
        ]
        if missing_fields:
            gaps.append(
                f"Missing required fields: {', '.join(missing_fields)}")

        # Check data quality scores
        satisfaction_scores = []
        for record in data_sample:
            if '整體滿意度' in record:
                satisfaction_scores.append(record['整體滿意度'])

        if satisfaction_scores:
            avg_score = sum(satisfaction_scores) / len(satisfaction_scores)
            if avg_score < 3.0:  # Assuming 1-5 scale
                gaps.append("Low satisfaction scores")
    else:
        # Fallback to old logic
        columns = metadata.get('columns', [])
        if len(columns) < 3:
            gaps.append("Insufficient data columns")

        individual_scores = metadata.get('individual_scores', [])
        if individual_scores:
            avg_score = sum(individual_scores) / len(individual_scores)
            if avg_score < 0.5:
                gaps.append("Low data quality scores")

    return gaps


def determine_document_status_from_metadata(metadata: Dict[str, Any]) -> str:
    """Determine document status based on expire date."""
    from datetime import datetime, date

    data_sample = metadata.get('data_sample', [])

    if data_sample and len(data_sample) > 0:
        # Check expire date from the first record
        first_record = data_sample[0]
        expire_date_str = first_record.get('expire_date')

        if expire_date_str:
            try:
                # Parse the expire date (assuming format like "2026-07-05")
                expire_date = datetime.strptime(expire_date_str,
                                                '%Y-%m-%d').date()
                today = date.today()

                if expire_date >= today:
                    return "有效"
                else:
                    return "過期"
            except (ValueError, TypeError):
                # If date parsing fails, return default
                return "有效"

    # Default to valid if no expire date found
    return "有效"


# def calculate_numeric_gap(query: str, result: Dict[str, Any]) -> NumericGap:
#     """Calculate numeric gaps for lead time, quality, and capacity."""
#     numeric_gap = NumericGap()

#     # Extract numbers from query
#     query_numbers = re.findall(r'\d+', query)

#     if not query_numbers:
#         return numeric_gap

#     # Check for lead time mentions
#     if any(word in query.lower()
#            for word in ['交期', 'lead', 'time', '天', 'days']):
#         # Look for lead time in data_sample
#         metadata = result.get('metadata', {})
#         data_sample = metadata.get('data_sample', [])

#         if data_sample and len(data_sample) > 0:
#             first_record = data_sample[0]
#             if '交期-準時率' in first_record:
#                 try:
#                     actual_lead_time = int(first_record['交期-準時率'])
#                     query_lead_time = int(query_numbers[0])
#                     gap = actual_lead_time - query_lead_time
#                     if gap > 0:
#                         numeric_gap.lead_time = f"+{gap} days"
#                     elif gap < 0:
#                         numeric_gap.lead_time = f"{gap} days"
#                 except:
#                     pass

#     # Check for quality mentions
#     if any(word in query.lower() for word in ['品質', 'quality', '分', 'score']):
#         metadata = result.get('metadata', {})
#         data_sample = metadata.get('data_sample', [])

#         if data_sample and len(data_sample) > 0:
#             first_record = data_sample[0]
#             if '品質-符合規範' in first_record:
#                 try:
#                     actual_quality = int(first_record['品質-符合規範'])
#                     query_quality = int(query_numbers[0])
#                     gap = actual_quality - query_quality
#                     if gap > 0:
#                         numeric_gap.quality = f"+{gap} points"
#                     elif gap < 0:
#                         numeric_gap.quality = f"{gap} points"
#                 except:
#                     pass

#     # Check for capacity mentions

# #     if any(word in query.lower()
# #            for word in ['容量', 'capacity', '數量', 'quantity']):
# #         metadata = result.get('metadata', {})
# #         if 'data_sample' in metadata:
# #             # Look for quantity in data sample
# #             for sample in metadata['data_sample']:
# #                 if 'quantity' in sample:
# #                     try:
# #                         actual_capacity = int(sample['quantity'])
# #                         query_capacity = int(query_numbers[0])
# #                         gap = actual_capacity - query_capacity
# #                         if gap > 0:
# #                             numeric_gap.capacity = f"+{gap} units"
# #                         elif gap < 0:
# #                             numeric_gap.capacity = f"{gap} units"
# #                         break
# #                     except:
# #                         pass

# # Check for capacity mentions
#     if any(word in query.lower()
#            for word in ['容量', 'capacity', '數量', 'quantity']):
#         metadata = result.get('metadata', {})
#         data_sample = metadata.get('data_sample', [])

#         if data_sample and len(data_sample) > 0:
#             for record in data_sample:
#                 if 'quantity' in record:
#                     try:
#                         actual_capacity = int(record['quantity'])
#                         query_capacity = int(query_numbers[0])
#                         gap = actual_capacity - query_capacity
#                         if gap > 0:
#                             numeric_gap.capacity = f"+{gap} units"
#                         elif gap < 0:
#                             numeric_gap.capacity = f"{gap} units"
#                         break
#                     except:
#                         pass

#     return numeric_gap


async def generate_embedding(text: str) -> List[float]:
    """Generate embedding for the given text using OpenAI API."""
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = client.embeddings.create(model="text-embedding-3-small",
                                            input=text)

        return response.data[0].embedding

    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Failed to generate embedding: {str(e)}")


# @router.get("/search/suggestions")
# async def get_search_suggestions(
#         query: str = Query(..., description="Partial query for suggestions"),
#         limit: int = Query(5, description="Number of suggestions to return")):
#     """Get search suggestions based on partial query."""
#     try:
#         # This could be enhanced with actual suggestion logic
#         suggestions = [
#             f"{query} 公司", f"{query} 產品", f"{query} 供應商", f"{query} 評鑑",
#             f"{query} 產業"
#         ]

#         return {"query": query, "suggestions": suggestions[:limit]}

#     except Exception as e:
#         raise HTTPException(status_code=500,
#                             detail=f"Failed to get suggestions: {str(e)}")


@router.get("/search/history")
async def get_search_history(limit: int = Query(
    10, description="Number of recent searches to return")):
    """Get recent search history."""
    try:
        async for session in get_session():
            from database.database import SearchQuery
            from sqlalchemy import desc

            # Get recent search queries
            stmt = select(SearchQuery).order_by(desc(
                SearchQuery.created_at)).limit(limit)
            result = await session.execute(stmt)
            queries = result.scalars().all()

            return {
                "queries": [{
                    "id": query.id,
                    "query_text": query.query_text,
                    "filters": query.filters,
                    "top_k": query.top_k,
                    "created_at": query.created_at
                } for query in queries]
            }
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Failed to get search history: {str(e)}")


@router.get("/search/results/{query_id}")
async def get_search_results(query_id: str):
    """Get results for a specific search query."""
    try:
        async for session in get_session():
            from database.database import SearchQuery, SearchResult

            # Get the query
            query_stmt = select(SearchQuery).where(SearchQuery.id == query_id)
            query_result = await session.execute(query_stmt)
            query = query_result.scalar_one_or_none()

            if not query:
                raise HTTPException(status_code=404, detail="Query not found")

            # Get the results
            results_stmt = select(SearchResult).where(
                SearchResult.query_id == query_id).order_by(SearchResult.rank)
            results_result = await session.execute(results_stmt)
            results = results_result.scalars().all()

            return {
                "query": {
                    "id": query.id,
                    "query_text": query.query_text,
                    "filters": query.filters,
                    "top_k": query.top_k,
                    "created_at": query.created_at
                },
                "results": [{
                    "id": result.id,
                    "company": result.company,
                    "product": result.product,
                    "completeness_score": result.completeness_score,
                    "semantic_score": result.semantic_score,
                    "doc_status": result.doc_status,
                    "total_score": result.total_score,
                    "rank": result.rank,
                    "vector_id": result.vector_id
                } for result in results]
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Failed to get search results: {str(e)}")
