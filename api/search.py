import numpy as np
from fastapi import APIRouter, HTTPException, Query
from database.database import VectorDB, get_session, SearchQuery
from database.database import SearchResult as ORMSearchResult
from database.schemas import SearchRequest, SearchResult as SearchResultSchema
from sqlalchemy import select, or_
from typing import List, Dict, Any
import openai
import os
from dotenv import load_dotenv
import re
import uuid
from datetime import datetime, date
import pandas as pd

load_dotenv()

router = APIRouter()

# A basic English + Chinese stopword set; extend with domain-specific items as needed
BASIC_STOPWORDS = set([
    # English
    'the',
    'a',
    'an',
    'and',
    'or',
    'but',
    'if',
    'then',
    'than',
    'that',
    'this',
    'those',
    'these',
    'i',
    'you',
    'he',
    'she',
    'it',
    'we',
    'they',
    'me',
    'my',
    'your',
    'his',
    'her',
    'our',
    'their',
    'is',
    'are',
    'was',
    'were',
    'be',
    'been',
    'being',
    'am',
    'do',
    'does',
    'did',
    'doing',
    'to',
    'for',
    'of',
    'on',
    'in',
    'with',
    'at',
    'by',
    'from',
    'as',
    'about',
    'into',
    'over',
    'after',
    'can',
    'could',
    'should',
    'would',
    'will',
    'shall',
    'may',
    'might',
    'must',
    'need',
    'needs',
    'needed',
    'want',
    'wants',
    'wanted',
    'looking',
    'look',
    'find',
    'get',
    'have',
    'has',
    'had',
    'please',
    'pls',
    'high',
    'quality',
    # Chinese (common function words)
    '的',
    '了',
    '在',
    '是',
    '和',
    '與',
    '及',
    '或',
    '而',
    '並',
    '就',
    '都',
    '很',
    '還',
    '又',
    '也',
    '我',
    '你',
    '他',
    '她',
    '它',
    '我們',
    '你們',
    '他們',
    '這',
    '那',
    '這些',
    '那些',
    '哪',
    '哪些',
    '請',
    '請問',
    '幫我',
    '找到',
    '尋找',
    '查找',
    '需要',
    '想要',
    '關於'
])

SIMPLE_SYNONYMS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
    "hundred": "100",
    "thousand": "1000",
    # Chinese numerals (basic single-character forms)
    "零": "0",
    "一": "1",
    "二": "2",
    "兩": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
    "十": "10",
    "百": "100",
    "千": "1000",
    "萬": "10000",
    "万": "10000",
}

METRIC_SYNONYMS = {
    "qty": "quantity",
    "數量": "quantity",
    "量": "quantity",
    "品質": "整體滿意度",
    "質量": "整體滿意度",
    "quality": "整體滿意度",
    "score": "整體滿意度",
}

# Ordinal words to numbers (EN + basic ZH via regex handled separately)
ORDINAL_WORDS = {
    'fs': 1,
    '1fs': 1,
    '2fs': 2,
    '3fs': 3,
    '4fs': 4,
    '5fs': 5,
    '6fs': 6,
    '7fs': 7,
    '8fs': 8,
    '9fs': 9,
    '10fs': 10,
    'first': 1,
    '1st': 1,
    'second': 2,
    '2nd': 2,
    'third': 3,
    '3rd': 3,
    'fourth': 4,
    '4th': 4,
    'fifth': 5,
    '5th': 5,
    'sixth': 6,
    '6th': 6,
    'seventh': 7,
    '7th': 7,
    'eighth': 8,
    '8th': 8,
    'ninth': 9,
    '9th': 9,
    'tenth': 10,
    '10th': 10,
}

# Mode synonyms for metric intent detection (EN + ZH)
MODE_SYNONYMS = {
    'max': [
        'highest', 'maximum', 'max', 'maximize', 'top', 'best', 'largest',
        'high', 'greatest', 'most', 'upper', 'peak', 'optimize', 'optimal',
        'maximal', '最高', '最大', '最多', '最佳', '最優', '最优', '最強', '最强', '高'
    ],
    'min': [
        'lowest', 'minimum', 'min', 'minimize', 'least', 'smallest', 'minimal',
        'low', 'lower', 'worst', 'bottom', 'mininal', 'minumal', '最低', '最小',
        '最少', '最差', '最弱', '低'
    ],
}

# Country alias normalization for fuzzy matching (codes, English, and common CJK)
COUNTRY_ALIAS_TO_CANON = {
    # Japan
    "jp": "jp",
    "jpn": "jp",
    "japan": "jp",
    "日本": "jp",
    "日本國": "jp",
    "日本国": "jp",
    # China (accept "ch" as an alias though ISO is "cn")
    "cn": "cn",
    "chn": "cn",
    "china": "cn",
    "ch": "cn",
    "中國": "cn",
    "中国": "cn",
    # Taiwan
    "tw": "tw",
    "twn": "tw",
    "taiwan": "tw",
    "台灣": "tw",
    "台湾": "tw",
    # Korea (South)
    "kr": "kr",
    "kor": "kr",
    "korea": "kr",
    "south korea": "kr",
    "ko": "kr",
    "韓國": "kr",
    "韩国": "kr",
    # United States
    "us": "us",
    "usa": "us",
    "united states": "us",
    "america": "us",
    # United Kingdom
    "uk": "uk",
    "gb": "uk",
    "gbr": "uk",
    "united kingdom": "uk",
    "great britain": "uk",
    "england": "uk",
    # Vietnam
    "vn": "vn",
    "vnm": "vn",
    "vietnam": "vn",
    "越南": "vn",
    # Thailand
    "th": "th",
    "tha": "th",
    "thailand": "th",
    "泰國": "th",
}

CANON_TO_ALIASES = {}
for alias, canon in COUNTRY_ALIAS_TO_CANON.items():
    CANON_TO_ALIASES.setdefault(canon, set()).add(alias)


def _canonical_country_token(token: str) -> str | None:
    try:
        t = (token or "").strip().lower()
    except Exception:
        t = ""
    if not t:
        return None
    return COUNTRY_ALIAS_TO_CANON.get(t, t)


def _expand_country_synonyms(token: str) -> list[str]:
    canon = _canonical_country_token(token)
    if not canon:
        return []
    aliases = CANON_TO_ALIASES.get(canon, set())
    # include the canonical itself as well
    out = set(aliases)
    out.add(canon)
    return sorted(out)


def _detect_metric_intent(query: str) -> Dict[str, Any] | None:
    """Detect metric intent from query using broad EN/ZH synonyms and both word orders.

    Supports patterns like:
    - "highest quality", "maximum tensile strength"
    - "quality highest", "品質 最高", "最高品質", "品質最高"
    """
    try:
        # Prepare alternations
        max_alt = "|".join(
            sorted(set(MODE_SYNONYMS['max']), key=len, reverse=True))
        min_alt = "|".join(
            sorted(set(MODE_SYNONYMS['min']), key=len, reverse=True))

        # Helper to normalize metric token
        def norm_metric(metric_raw: str) -> tuple[str, str]:
            mr = (metric_raw or '').strip()
            token = mr.split()[0].lower() if mr else ''
            token = METRIC_SYNONYMS.get(token, token)
            return token, mr.lower()

        # 1) Mode first (max)
        m = re.search(
            rf"(?:{max_alt})(?:的)?\s*([A-Za-z\u4e00-\u9fff_][A-Za-z0-9 \u4e00-\u9fff_\-]*)",
            query, re.IGNORECASE)
        if m:
            metric_token, metric_raw = norm_metric(m.group(1))
            if metric_token:
                rank = 1
                # detect ordinals around the phrase
                ql = query.lower()
                for ow, rv in ORDINAL_WORDS.items():
                    if ow in ql:
                        rank = rv
                        break
                # Chinese: 第N (e.g., 第2, 第二)
                m_ord = re.search(r"第\s*(\d+)", query)
                if m_ord:
                    try:
                        rank = int(m_ord.group(1))
                    except Exception:
                        pass
                return {
                    'mode': 'max',
                    'metric': metric_token,
                    'metric_raw': metric_raw,
                    'rank': rank
                }

        # 2) Mode first (min)
        m = re.search(
            rf"(?:{min_alt})(?:的)?\s*([A-Za-z\u4e00-\u9fff_][A-Za-z0-9 \u4e00-\u9fff_\-]*)",
            query, re.IGNORECASE)
        if m:
            metric_token, metric_raw = norm_metric(m.group(1))
            if metric_token:
                rank = 1
                ql = query.lower()
                for ow, rv in ORDINAL_WORDS.items():
                    if ow in ql:
                        rank = rv
                        break
                m_ord = re.search(r"第\s*(\d+)", query)
                if m_ord:
                    try:
                        rank = int(m_ord.group(1))
                    except Exception:
                        pass
                return {
                    'mode': 'min',
                    'metric': metric_token,
                    'metric_raw': metric_raw,
                    'rank': rank
                }

        # 3) Metric first (max at end)
        m = re.search(
            rf"([A-Za-z\u4e00-\u9fff_][A-Za-z0-9 \u4e00-\u9fff_\-]*)\s*(?:的)?\s*(?:{max_alt})",
            query, re.IGNORECASE)
        if m:
            metric_token, metric_raw = norm_metric(m.group(1))
            if metric_token:
                rank = 1
                ql = query.lower()
                for ow, rv in ORDINAL_WORDS.items():
                    if ow in ql:
                        rank = rv
                        break
                m_ord = re.search(r"第\s*(\d+)", query)
                if m_ord:
                    try:
                        rank = int(m_ord.group(1))
                    except Exception:
                        pass
                return {
                    'mode': 'max',
                    'metric': metric_token,
                    'metric_raw': metric_raw,
                    'rank': rank
                }

        # 4) Metric first (min at end)
        m = re.search(
            rf"([A-Za-z\u4e00-\u9fff_][A-Za-z0-9 \u4e00-\u9fff_\-]*)\s*(?:的)?\s*(?:{min_alt})",
            query, re.IGNORECASE)
        if m:
            metric_token, metric_raw = norm_metric(m.group(1))
            if metric_token:
                rank = 1
                ql = query.lower()
                for ow, rv in ORDINAL_WORDS.items():
                    if ow in ql:
                        rank = rv
                        break
                m_ord = re.search(r"第\s*(\d+)", query)
                if m_ord:
                    try:
                        rank = int(m_ord.group(1))
                    except Exception:
                        pass
                return {
                    'mode': 'min',
                    'metric': metric_token,
                    'metric_raw': metric_raw,
                    'rank': rank
                }

    except Exception:
        pass

    return None


def preprocess_query_extract_keywords(query: str) -> Dict[str, Any]:
    """Extract product codes and main keywords for embedding.

    - Detect product-like codes (e.g., Q001, Q02, ABC-123)
    - Keep only meaningful alphanumeric tokens for embedding
    """
    if not query:
        return {"keywords_text": "", "product_code": None}

    # Detect product code (more flexible patterns)
    product_code = None
    try:
        # Match patterns like: Q2, Q02, Q-2, Q-02, ABC2, ABC-2, etc.
        code_match = re.search(r"\b([A-Za-z]+-?[0-9]+|[A-Za-z]?[0-9]+)\b",
                               query, re.IGNORECASE)
        if code_match:
            raw_code = code_match.group(1).upper().replace('-', '')
            # Only consider it a product code if it has both letters and numbers
            if re.search(r'[A-Z]', raw_code) and re.search(r'[0-9]', raw_code):
                product_code = raw_code
    except Exception:
        product_code = None

    # Tokenize: capture ASCII words/numbers or contiguous CJK blocks
    tokens = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", query.lower())
    filtered: List[str] = []
    for t in tokens:
        # synonym normalization (e.g., two -> 2)
        t = SIMPLE_SYNONYMS.get(t, t)
        if t in BASIC_STOPWORDS:
            continue
        if t.isdigit() and len(t) < 2:
            continue
        # For ASCII terms, skip too-short tokens; keep CJK tokens as-is
        if re.match(r"^[A-Za-z0-9]+$", t) and len(t) < 2:
            continue
        filtered.append(t)

    # Include literal product code to influence embedding if present
    if product_code and product_code.lower() not in filtered:
        filtered.append(product_code.lower())

    keywords_text = " ".join(filtered).strip() or query
    # Enhanced metric intent detection with EN/ZH synonyms and flexible order
    metric_intent = _detect_metric_intent(query)

    return {
        "keywords_text": keywords_text,
        "product_code": product_code,
        "metric_intent": metric_intent
    }


def _normalize_product_code(code: str) -> str:
    """Normalize product codes for comparison: uppercase, remove '-', drop leading zeros in numeric suffix.

    Examples:
    - Q02 -> Q2; Q002 -> Q2; Q2 -> Q2; ABC-001 -> ABC1
    """
    if code is None:
        return ""
    raw = str(code).upper().replace('-', '')
    # Split alpha prefix and numeric suffix
    m = re.match(r"^([A-Z]*)(\d+)$", raw)
    if not m:
        return raw
    prefix, digits = m.groups()
    digits_no_zeros = digits.lstrip('0') or '0'
    return f"{prefix}{digits_no_zeros}"


def parse_filters_string(filters_str: str) -> Dict[str, List[str]]:
    """
    Parse filters string into dictionary format.
    Simple format: "扣件" -> {"industry_category": ["扣件"]}
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
                    # Normalize aliases
                    kl = key.lower()
                    if kl in ["category", "industry_category", "industry"]:
                        key = "industry_category"
                    elif kl in [
                            "country", "nation", "國家", "國別", "address", "addr",
                            "location", "location_addr"
                    ]:
                        key = "country"
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
    - Multi-factor sorting (Completeness + SemanticSim)
    """
    try:
        # Extract parameters from request
        query = request.query_text  # Prefer new k_top, then legacy t_top, then default top_k
        filters_str = getattr(request, 'filters', None)
        top_k = (getattr(request, 'k_top', None)
                 if hasattr(request, 'k_top') else None)
        if top_k is None:
            top_k = (getattr(request, 't_top', None) if hasattr(
                request, 't_top') else None)
        if top_k is None:
            top_k = request.top_k
        # Guard against non-positive values
        try:
            if int(top_k) <= 0:
                top_k = request.top_k
        except Exception:
            top_k = request.top_k
        print(f"Query: {query}")
        # Build filters from new input style first, fallback to legacy string
        filters: Dict[str, List[str]] = {}
        try:
            # Primary fields
            ind = getattr(request, 'industry', None)
            ctry = getattr(request, 'country', None)
            if ind:
                inds = ind if isinstance(ind, list) else [ind]
                filters['industry_category'] = [
                    str(c).strip() for c in inds if str(c).strip()
                ]
            if ctry:
                ctrs = ctry if isinstance(ctry, list) else [ctry]
                filters['country'] = [
                    str(c).strip() for c in ctrs if str(c).strip()
                ]

        except Exception:
            filters = {}

        # Enforce both category and country must be provided
        industry_vals = filters.get("industry_category") or filters.get(
            "industry") or []
        country_vals = filters.get("country") or []
        if not industry_vals or not country_vals:
            raise HTTPException(
                status_code=400,
                detail=
                "Both category and country filters are required (e.g., category:XXX;country:YYY)"
            )

        # Preprocess query to extract keywords and detect product code
        qp = preprocess_query_extract_keywords(query)
        product_code = qp.get("product_code")
        query_for_embedding = qp.get("keywords_text") or query
        metric_intent = qp.get("metric_intent")
        # Company list intent: list all companies in current filter context
        company_list_intent = False
        try:
            if re.search(
                    r"\b(list|show)\b.*\b(companies|company|suppliers?)\b",
                    query, re.IGNORECASE):
                company_list_intent = True
            if re.search(r"(列出|清單|所有).*(公司|廠商)", query):
                company_list_intent = True
            if re.fullmatch(r"\s*(list\s*all)\s*", query, re.IGNORECASE):
                company_list_intent = True
            if re.fullmatch(r"\s*(all\s*list)\s*", query, re.IGNORECASE):
                company_list_intent = True
        except Exception:
            company_list_intent = False

        # Step 1: Filter VectorDB by filter values first
        filtered_groups = await filter_vectordb_by_filters(filters)

        if not filtered_groups:
            return {"results": []}

        # Step 2: If listing companies, aggregate without embedding
        if company_list_intent:
            # Aggregate across filtered groups
            seen: Dict[str, str] = {}

            def add_company(name: str):
                if not name:
                    return
                norm = str(name).strip()
                if not norm or norm.lower() == 'unknown company':
                    return
                norm = re.sub(r"\s+", " ", norm)
                key = norm.lower()
                if key not in seen:
                    seen[key] = norm  # preserve first-seen casing

            for vector_entry in filtered_groups:
                md = vector_entry.metadata_json or {}
                # product_metrics preferred
                for m in (md.get('product_metrics') or []):
                    add_company(m.get('company'))
                # data_sample fallback
                for rec in (md.get('data_sample') or []):
                    add_company(
                        rec.get('客戶名稱') or rec.get('公司名稱')
                        or rec.get('company_name') or rec.get('company'))
                # direct metadata fallback
                add_company(
                    md.get('客戶名稱') or md.get('公司名稱') or md.get('company_name')
                    or md.get('company'))

            companies = sorted(seen.values(), key=lambda s: s.lower())
            return {"companies": companies}

        # Step 2.5: Intent - show all information about X (company or product)
        show_all_intent = False
        target_name = None
        try:
            m = re.search(r"show\s+all\s+the\s+information\s+about\s+(.+)$",
                          query, re.IGNORECASE)
            if not m:
                m = re.search(
                    r"(顯示|查看|列出).*(全部|所有).*(資訊|信息|資料).*(關於|關於)?\s*([\w\u4e00-\u9fff\- ]+)",
                    query)
                if m:
                    target_name = m.group(len(m.groups()))
            else:
                target_name = m.group(1)
            if target_name:
                target_name = target_name.strip().strip('"\'')
                show_all_intent = True
        except Exception:
            show_all_intent = False

        # If show-all intent, return full records for matching product/company within filtered groups
        if show_all_intent and target_name:
            target_l = target_name.lower()
            matches: List[Dict[str, Any]] = []
            for ve in filtered_groups:
                md = ve.metadata_json or {}
                # search product_metrics first
                for mrec in (md.get('product_metrics') or []):
                    pid = str(mrec.get('product_id') or '').lower()
                    cname = str(mrec.get('company') or '').lower()
                    if target_l in pid or target_l in cname:
                        matches.append({
                            "vector_id": ve.id,
                            "industry": ve.filter,
                            "record": mrec
                        })
                # also check data_sample
                for rec in (md.get('data_sample') or []):
                    rec_s = " ".join([
                        str(v) for v in rec.values() if v is not None
                    ]).lower()
                    if target_l and target_l in rec_s:
                        matches.append({
                            "vector_id": ve.id,
                            "industry": ve.filter,
                            "record": rec
                        })

            if not matches:
                raise HTTPException(
                    status_code=404,
                    detail=
                    "No matching product or company found for show-all request within the selected category"
                )

            return {"matches": matches}

        # Step 3: Generate embedding using preprocessed keywords
        query_embedding = await generate_embedding(query_for_embedding)

        # Step 4: Perform semantic search within filtered groups
        vector_results = await semantic_search_within_groups(
            query, query_embedding, filtered_groups, top_k, product_code,
            metric_intent, filters)

        # Step 5: Convert to response format
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

            # If a target product code is detected, prefer the matched product/company from metadata
            preferred_product = result.get('matched_product') or result.get(
                'product_name')
            preferred_company = result.get('matched_company') or company_name

            formatted_result = SearchResultSchema(
                company=preferred_company,
                product=preferred_product,
                completeness_score=int(result['completeness_score'] * 100),
                semantic_score=round(result['semantic_similarity'], 2),
                # numeric_gap=calculate_numeric_gap(query, result),
                doc_status=result['document_status'],
                total_score=int(result['overall_score'] * 100))
            formatted_results.append(formatted_result)

            # Collect vector IDs for database storage
            vector_ids.append(result['metadata']['vector_id'])

        # Save query and results to database
        query_id = None
        result_ids: List[str] = []
        try:
            query_id, result_ids = await save_search_query_and_results(
                query_text=query,
                filters=filters_str,
                top_k=top_k,
                results=formatted_results,
                vector_ids=vector_ids)
            print(f"Search query saved with ID: {query_id}")
        except Exception as e:
            print(f"Failed to save search query: {e}")
            # Continue even if saving fails

        # Return results and query_id (include requested top_k and actual count)
        return {
            "query_id": query_id,
            "top_k": top_k,
            "returned": len(formatted_results),
            "result_ids": result_ids,
            "results": formatted_results
        }

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
                    country_filters = filters.get("country", [])

                    # If both industry and country are provided, narrow by both
                    if industry_filters and country_filters:
                        industry_conditions = [
                            VectorDB.filter.ilike(f"%{industry}%")
                            for industry in industry_filters
                        ]
                        country_conditions = [
                            # match the pipe-delimited country segment if present
                            VectorDB.filter.ilike(f"%|{country}%")
                            for country in country_filters
                        ]
                        stmt = stmt.where(or_(*industry_conditions)).where(
                            or_(*country_conditions))
                    elif industry_filters:
                        industry_conditions = [
                            VectorDB.filter.ilike(f"%{industry}%")
                            for industry in industry_filters
                        ]
                        stmt = stmt.where(or_(*industry_conditions))
                    elif country_filters:
                        country_conditions = []
                        for country in country_filters:
                            for syn in _expand_country_synonyms(country):
                                country_conditions.append(
                                    VectorDB.filter.ilike(f"%|{syn}%"))
                        if country_conditions:
                            stmt = stmt.where(or_(*country_conditions))

                    # Execute query
                    result = await session.execute(stmt)
                    filtered_entries = result.scalars().all()
                    return filtered_entries  # Return immediately on success

                except Exception as e:
                    await session.rollback()
                    print(f"Database query error (attempt {attempt + 1}): {e}")

                    # If it's a cached statement error, clear cache and retry
                    # if "InvalidCachedStatementError" in str(e):
                    #     await clear_statement_cache()
                    #     break  # Break out of session loop to retry
                    # else:
                    #     raise e  # Re-raise if it's a different error

        except Exception as e:
            print(f"Session error (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                print("Max retries reached, returning empty list")
                return []
            # Wait a bit before retrying
            import asyncio
            await asyncio.sleep(0.1 * (attempt + 1))

    return filtered_entries


async def semantic_search_within_groups(
        query: str,
        query_embedding: List[float],
        filtered_groups: List[VectorDB],
        top_k: int,
        product_code: str = None,
        metric_intent: Dict[str, Any] = None,
        filters: Dict[str, List[str]] | None = None) -> List[Dict[str, Any]]:
    """
    Step 2: Perform semantic search within the filtered groups.
    Returns ranked results based on semantic similarity.
    """
    results = []

    # Calculate similarity for each filtered group
    country_filters = (filters or {}).get("country", [])
    country_norms = {
        c
        for c in (_canonical_country_token(x) for x in country_filters) if c
    }
    any_country_required = len(country_norms) > 0

    for vector_entry in filtered_groups:
        # Calculate cosine similarity
        similarity = calculate_cosine_similarity(query_embedding,
                                                 vector_entry.embedding)

        # Get metadata
        metadata = vector_entry.metadata_json or {}

        # Validate country presence inside this group if required
        if any_country_required:
            countries_md = set()
            for m in (metadata.get('product_metrics') or []):
                cval = m.get('country')
                if cval:
                    try:
                        cc = _canonical_country_token(cval)
                        if cc:
                            countries_md.add(cc)
                    except Exception:
                        pass
            for c in (metadata.get('countries') or []):
                try:
                    cc = _canonical_country_token(c)
                    if cc:
                        countries_md.add(cc)
                except Exception:
                    pass
            for rec in (metadata.get('data_sample') or []):
                for k, v in rec.items():
                    try:
                        kl = str(k).lower()
                        if any(x in kl for x in [
                                "country", "國家", "國別", "地址", "address", "addr",
                                "所在地"
                        ]):
                            s = _canonical_country_token(v)
                            if s:
                                countries_md.add(s)
                    except Exception:
                        continue
            if countries_md and not (countries_md & country_norms):
                # Skip this group entirely if it has country info and does not match
                # If no country info at all, we won't skip here; we'll validate later
                continue

        # Debug: Print metadata structure to understand the data
        print(
            f"Debug - Metadata keys in semantic: {list(metadata.keys()) if metadata else 'Empty metadata'}"
        )
        print(f"Debug - Filter value: {vector_entry.filter}")

        # Get completeness score for the group (will be overridden for individual products)
        # Use average_score as group-level completeness
        completeness = 0.5
        try:
            avg = metadata.get('average_score')
            if isinstance(avg, (int, float)):
                completeness = float(avg)
            else:
                indiv = metadata.get('individual_scores') or []
                indiv_floats = [
                    float(x) for x in indiv if isinstance(x, (int, float))
                ]
                if indiv_floats:
                    completeness = float(np.mean(indiv_floats))
        except Exception:
            completeness = 0.5

        # Determine document status (will be updated later with matched product)
        doc_status = determine_document_status_from_metadata(metadata)

        # Calculate overall score
        overall_score = (completeness * 0.54 + similarity * 0.3)
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

        # If still missing, try take first available product id from product_ids
        if not product_name:
            pid_list = metadata.get('product_ids') or []
            if pid_list:
                product_name = str(pid_list[0])

        matched_product = None
        matched_company = None

        # If a product code was extracted, see if it exists in metadata and boost score
        if product_code:
            meta_products = metadata.get('product_ids') or []
            # Normalize IDs for comparison (strip hyphens, upper, drop leading zeros)
            normalized = {
                _normalize_product_code(p): str(p)
                for p in meta_products
            }
            normalized_query = _normalize_product_code(product_code)
            if normalized_query in normalized:
                matched_product = normalized[normalized_query]
                # Try mapping to company
                p2c = metadata.get('product_to_company') or {}
                matched_company = p2c.get(matched_product)
                # Apply a boost to overall score to favor exact product matches
                overall_score += 0.2 * 0.4

        # If a metric intent is specified, pick best product in this group
        if metric_intent:
            metrics = metadata.get('product_metrics') or []
            if metrics:
                metric_key = metric_intent.get('metric')
                metric_raw = metric_intent.get('metric_raw') or metric_key

                # Build candidate keys: normalized token, raw phrase, and variants
                candidates = []
                if metric_key:
                    candidates.append(metric_key)
                if metric_raw:
                    candidates.append(metric_raw)
                    candidates.extend(metric_raw.replace('-', ' ').split())

                def resolve_value(m: Dict[str, Any]) -> Any:
                    # Direct top-level numeric
                    for c in candidates:
                        v = m.get(c)
                        if isinstance(v, (int, float)):
                            return v
                    # Look into fields dict with case-insensitive and partial match
                    fields = m.get('fields') or {}
                    lowered = {str(k).lower(): v for k, v in fields.items()}
                    for c in candidates:
                        c_low = str(c).lower()
                        # exact
                        if c_low in lowered and isinstance(
                                lowered[c_low], (int, float)):
                            return lowered[c_low]
                        # partial startswith
                        for k, v in lowered.items():
                            if isinstance(
                                    v, (int, float)) and (k.startswith(c_low)
                                                          or c_low in k):
                                return v
                    return None

                # If a product_code was provided but didn't resolve to an exact
                # match above, treat it as a prefix filter (e.g., "Q012*"), and
                # restrict metric aggregation to those products first.
                filtered_metrics = metrics
                if product_code:
                    try:
                        normalized_query = _normalize_product_code(
                            product_code)
                        prefixed = []
                        for m in metrics:
                            pid = m.get('product_id') or m.get('product')
                            if pid and _normalize_product_code(pid).startswith(
                                    normalized_query):
                                prefixed.append(m)
                        if prefixed:
                            filtered_metrics = prefixed
                    except Exception:
                        # On any error, fall back to original metrics list
                        filtered_metrics = metrics

                enriched = []
                for m in filtered_metrics:
                    val = resolve_value(m)
                    if val is not None:
                        enriched.append((m, float(val)))

                if enriched:
                    reverse = metric_intent.get('mode') == 'max'
                    rank = max(1, int(metric_intent.get('rank') or 1))
                    sorted_list = sorted(enriched,
                                         key=lambda t: t[1],
                                         reverse=reverse)
                    if rank <= len(sorted_list):
                        chosen = sorted_list[rank - 1][0]
                    else:
                        chosen = sorted_list[-1][0]
                    matched_product = chosen.get(
                        'product_id') or matched_product
                    matched_company = chosen.get('company') or matched_company
                    # slight boost when honoring metric intent rank
                    overall_score += 0.15 * 0.4

        # Update document status based on matched product if available
        if matched_product:
            doc_status = determine_document_status_from_metadata(
                metadata, matched_product)

        results.append({
            "company_name": company_name,
            "product_name": product_name,
            "document_status": doc_status,
            "completeness_score": completeness,
            "semantic_similarity": similarity,
            # "numeric_fit": numeric_fit,
            "overall_score": overall_score,
            "matched_product": matched_product,
            "matched_company": matched_company,
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
                "product_metrics":
                metadata.get('product_metrics', []),
                "filter":
                vector_entry.filter,
                "countries":
                metadata.get('countries', [])
            }
        })

    # Sort by overall score (descending)
    results.sort(key=lambda x: x['overall_score'], reverse=True)

    # Expand to product-level items if we matched specific products or have metrics
    expanded: List[Dict[str, Any]] = []
    for r in results:
        pm = r.get('metadata', {}).get('product_metrics') or []
        if r.get('matched_product'):
            # Only include the matched product from this group
            matched_id = r['matched_product']
            # Find company if possible
            matched_company = r.get('matched_company') or r.get('company_name')
            expanded.append({
                "company_name":
                matched_company or r.get('company_name'),
                "product_name":
                matched_id,
                "document_status":
                r['document_status'],
                "completeness_score":
                r['completeness_score'],
                "semantic_similarity":
                r['semantic_similarity'],
                "overall_score":
                r['overall_score'] +
                0.05 * 0.4,  # slight preference to direct match
                "metadata":
                r["metadata"],
            })
        elif pm:
            # Add products from this group by quality_score to diversify
            # Use all available products, not limited to 3
            top_products = sorted(pm,
                                  key=lambda m: (m.get('quality_score') or 0),
                                  reverse=True)[:top_k]
            for m in top_products:
                # Get individual product completeness score
                item_completeness = m.get('score') if isinstance(
                    m.get('score'), (int, float)) else r['completeness_score']

                # Calculate individual product document status
                product_metadata = {
                    'product_metrics': [m],
                    'data_sample': [
                        rec for rec in (metadata.get('data_sample') or [])
                        if (rec.get('問卷編號') or rec.get('product_id')
                            or rec.get('product')) == m.get('product_id')
                    ]
                }
                product_doc_status = determine_document_status_from_metadata(
                    product_metadata)

                # Calculate individual product overall score
                # Use group semantic similarity but individual completeness
                individual_overall_score = (item_completeness * 0.54 +
                                            r['semantic_similarity'] * 0.3)

                expanded.append({
                    "company_name":
                    m.get('company') or r.get('company_name'),
                    "product_name":
                    m.get('product_id') or r.get('product_name'),
                    "document_status":
                    product_doc_status,  # Individual product status
                    "completeness_score":
                    item_completeness,  # Individual product completeness
                    "semantic_similarity":
                    r['semantic_similarity'],  # Group semantic similarity
                    "overall_score":
                    individual_overall_score,  # Individual product overall score
                    "metadata":
                    r["metadata"],
                })
        else:
            expanded.append(r)

    # De-duplicate by (company_name, product_name)
    seen_pairs = set()
    deduped = []
    for item in expanded:
        pair = (item.get('company_name'), item.get('product_name'))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        deduped.append(item)

    # Sort again after expansion
    deduped.sort(key=lambda x: x['overall_score'], reverse=True)

    # Apply country filter at result level and enforce validation if requested
    if any_country_required:
        filtered = []
        for item in deduped:
            md = item.get('metadata', {})
            in_country = False
            pm = md.get('product_metrics') or []
            pname = item.get('product_name')
            for m in pm:
                try:
                    if (m.get('product_id') == pname
                        ) and m.get('country') and _canonical_country_token(
                            m.get('country')) in country_norms:
                        in_country = True
                        break
                except Exception:
                    continue
            if not in_country:
                for c in (md.get('countries') or []):
                    try:
                        if _canonical_country_token(c) in country_norms:
                            in_country = True
                            break
                    except Exception:
                        continue
            if in_country:
                filtered.append(item)

        if not filtered:
            raise HTTPException(
                status_code=404,
                detail=
                "No data for requested country within the selected category")
        return filtered[:top_k]

    # Return only the top_k items
    return deduped[:top_k]


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


def determine_document_status_from_metadata(metadata: Dict[str, Any]) -> str:
    """Determine document status by scanning expire dates from products.
    
    Strategy:
    - Collect candidate dates from `product_metrics` and `data_sample`
    - Parse robustly (strings, Excel serials, datetime/date)
    - If ANY date is expired, return "過期", otherwise "有效"
    - Default to "過期" if nothing parseable is found
    """
    from datetime import datetime, date
    import re
    import pandas as pd

    def parse_any_date(value):
        try:
            # Normalize strings like "2025- 9-10"
            if isinstance(value, str):
                value = re.sub(r"\s+", " ", value.strip())
                value = value.replace("- ", "-").replace(" -", "-")
            # Already date/datetime
            if isinstance(value, datetime):
                return value.date()
            if isinstance(value, date):
                return value
            # Excel serial (numeric)
            if isinstance(value, (int, float)) and not pd.isna(value):
                excel_epoch = datetime(1899, 12, 30)
                if 1 <= float(value) <= 2958465:
                    return (excel_epoch +
                            pd.Timedelta(days=float(value))).date()
            # Pandas fallback
            ts = pd.to_datetime(str(value), errors='coerce')
            if pd.notna(ts) and getattr(ts, 'year', 0) > 1900:
                return ts.date()
        except Exception:
            return None
        return None

    def collect_candidate_dates(md: Dict[str, Any]) -> list[date]:
        candidates: list[date] = []
        keys = ['expire_date', '到期', '有效期', '截止', '截止日']

        print(f"Debug - Scanning products for expire dates...")

        # From product_metrics
        product_metrics = md.get('product_metrics', [])
        print(f"Debug - Checking {len(product_metrics)} product metrics")

        for i, pm in enumerate(product_metrics):
            print(f"Debug - Product metric {i} keys: {list(pm.keys())}")
            # top-level
            top = pm.get('expire_date')
            if top is not None:
                print(f"Debug - Found top-level expire_date: {top}")
                d = parse_any_date(top)
                if d:
                    candidates.append(d)
                    print(f"Debug - Parsed top-level date: {d}")
            # fields
            fields = pm.get('fields', {})
            print(f"Debug - Fields keys: {list(fields.keys())}")
            for k, v in fields.items():
                try:
                    if any(x in str(k).lower() for x in keys):
                        print(f"Debug - Found date field '{k}': {v}")
                        d = parse_any_date(v)
                        if d:
                            candidates.append(d)
                            print(f"Debug - Parsed field date: {d}")
                except Exception as e:
                    print(f"Debug - Error processing field {k}: {e}")
                    continue

        # From data_sample
        data_sample = md.get('data_sample', [])
        print(f"Debug - Checking {len(data_sample)} data sample records")

        for i, rec in enumerate(data_sample):
            print(f"Debug - Data sample {i} keys: {list(rec.keys())}")
            for k, v in (rec or {}).items():
                try:
                    if any(x in str(k).lower() for x in keys):
                        print(
                            f"Debug - Found date field in data_sample '{k}': {v}"
                        )
                        d = parse_any_date(v)
                        if d:
                            candidates.append(d)
                            print(f"Debug - Parsed data_sample date: {d}")
                except Exception as e:
                    print(
                        f"Debug - Error processing data_sample field {k}: {e}")
                    continue

        print(
            f"Debug - Collected {len(candidates)} candidate dates: {candidates}"
        )
        return candidates

    try:
        dates = collect_candidate_dates(metadata or {})
        if not dates:
            print("Debug - No valid dates found, defaulting to 有效")
            return "過期"

        today = date.today()
        print(f"Debug - Today: {today}")

        # Check each date individually
        valid_dates = []
        expired_dates = []

        for d in dates:
            if d >= today:
                valid_dates.append(d)
                print(f"Debug - Date {d} is valid (>= today)")
            else:
                expired_dates.append(d)
                print(f"Debug - Date {d} is expired (< today)")

        print(f"Debug - Valid dates: {valid_dates}")
        print(f"Debug - Expired dates: {expired_dates}")

        # If ANY date is expired, the document status is expired
        if expired_dates:
            print("Debug - Document status: 過期 (has expired dates)")
            return "過期"
        else:
            print("Debug - Document status: 有效 (all dates valid)")
            return "有效"

    except Exception as e:
        print(f"Debug - Exception in document status: {e}")
        import traceback
        traceback.print_exc()
        return "有效"


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
            from database.database import SearchQuery, SearchResult as ORMSearchResult

            # Get the query
            query_stmt = select(SearchQuery).where(SearchQuery.id == query_id)
            query_result = await session.execute(query_stmt)
            query = query_result.scalar_one_or_none()

            if not query:
                raise HTTPException(status_code=404, detail="Query not found")

            # Get the results
            results_stmt = select(ORMSearchResult).where(
                ORMSearchResult.query_id == query_id).order_by(
                    ORMSearchResult.rank)
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


async def save_search_query_and_results(
        query_text: str, filters: str, top_k: int, results: list,
        vector_ids: list) -> tuple[str, list[str]]:
    """Save search query and its results to the database.

    Returns (query_id, result_ids) where result_ids align with the input results order.
    """

    query_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()

    async for session in get_session():
        try:
            # Save the search query
            search_query = SearchQuery(id=query_id,
                                       query_text=query_text,
                                       filters=filters,
                                       top_k=top_k,
                                       created_at=created_at)
            session.add(search_query)

            # Save each result
            created_result_ids: list[str] = []
            for rank, (result,
                       vector_id) in enumerate(zip(results, vector_ids), 1):
                new_id = str(uuid.uuid4())
                search_result = ORMSearchResult(
                    id=new_id,
                    query_id=query_id,
                    company=result.company,
                    product=result.product,
                    completeness_score=result.completeness_score,
                    semantic_score=result.semantic_score,
                    doc_status=result.doc_status,
                    total_score=result.total_score,
                    rank=rank,
                    vector_id=vector_id)
                session.add(search_result)
                created_result_ids.append(new_id)

            await session.commit()
            return query_id, created_result_ids

        except Exception as e:
            await session.rollback()
            raise e
