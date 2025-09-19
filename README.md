# Semantic Search API

A FastAPI-based semantic search system for company and product data with AI-powered embeddings and intelligent filtering.

## Overview

This API provides semantic search capabilities for company and product data using OpenAI embeddings. It supports file uploads, semantic search with fuzzy matching, and feedback collection for continuous improvement.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, no authentication is required. All endpoints are publicly accessible.

## API Endpoints

### 1. Health Check Endpoints

#### GET `/`

**Description:** Basic health check endpoint  
**Response:**

```json
{
  "message": "Semantic Search API is running",
  "status": "healthy"
}
```

#### GET `/health`

**Description:** Detailed health check endpoint  
**Response:**

```json
{
  "status": "healthy",
  "service": "semantic_search_api",
  "version": "1.0.0"
}
```

---

### 2. File Upload Endpoints

#### POST `/api/upload`

**Description:** Upload Excel or CSV file containing company or product data  
**Content-Type:** `multipart/form-data`

**Request Body:**

- `file` (file, required): Excel (.xlsx, .xls) or CSV file

**File Requirements:**

- **Product Data:** Must contain an industry category column (e.g., '產業別', 'industry_category')
- **Company Data:** Must contain a column named "公司名稱" (Company Name)

**Response:**

```json
{
  "message": "Successfully processed X industry groups with embeddings",
  "file_id": "uuid-string",
  "groups_processed": 5,
  "data_type": "product"
}
```

**Error Responses:**

- `400`: Invalid file format (only CSV and Excel supported)
- `400`: Missing required columns
- `500`: File processing error

**Example cURL:**

```bash
curl -X POST "http://localhost:8000/api/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data.xlsx"
```

---

### 3. Search Endpoints

#### POST `/api/search`

**Description:** Perform semantic search with fuzzy matching and multi-factor sorting

**Request Body:**

```json
{
  "query_text": "string",
  "filters": "string",
  "top_k": 5
}
```

**Parameters:**

- `query_text` (string, required): Natural language search query
- `filters` (string, optional): Filter criteria in specific format
  - Simple format: `"扣件"` (treats as industry category)
  - Complex format: `"industry:FOOD,FOOD2;country:VN,TH"` (key:value pairs)
- `top_k` (integer, optional): Number of results to return (default: 5)

**Response:**

```json
{
  "results": [
    {
      "company": "Company Name",
      "product": "Product ID",
      "completeness_score": 85,
      "semantic_score": 0.92,
      "doc_status": "有效",
      "total_score": 88
    }
  ]
}
```

**Search Features:**

- Natural language queries
- Semantic similarity search
- Multi-factor sorting (Completeness + Semantic Similarity)
- Industry category filtering
- Country filtering (if available in metadata)

**Example cURL:**

```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "food processing equipment",
    "filters": "industry:FOOD",
    "top_k": 10
  }'
```

#### GET `/api/search/history`

**Description:** Get recent search history

**Query Parameters:**

- `limit` (integer, optional): Number of recent searches to return (default: 10)

**Response:**

```json
{
  "queries": [
    {
      "id": "query-uuid",
      "query_text": "search query",
      "filters": "industry:FOOD",
      "top_k": 5,
      "created_at": "2024-01-01T12:00:00"
    }
  ]
}
```

#### GET `/api/search/results/{query_id}`

**Description:** Get detailed results for a specific search query

**Path Parameters:**

- `query_id` (string, required): UUID of the search query

**Response:**

```json
{
  "query": {
    "id": "query-uuid",
    "query_text": "search query",
    "filters": "industry:FOOD",
    "top_k": 5,
    "created_at": "2024-01-01T12:00:00"
  },
  "results": [
    {
      "id": "result-uuid",
      "company": "Company Name",
      "product": "Product ID",
      "completeness_score": 85,
      "semantic_score": 0.92,
      "doc_status": "有效",
      "total_score": 88,
      "rank": 1,
      "vector_id": "vector-uuid"
    }
  ]
}
```

---

### 4. Feedback Endpoints

#### POST `/api/feedback`

**Description:** Submit feedback for a search result

**Request Body:**

```json
{
  "query_id": "string",
  "result_id": "string",
  "action_type": "string"
}
```

**Parameters:**

- `query_id` (string, required): UUID of the search query
- `result_id` (string, required): UUID of the search result
- `action_type` (string, required): Action type - one of:
  - `"keep"`: User wants to keep this result
  - `"reject"`: User wants to reject this result
  - `"compare"`: User wants to compare this result with others

**Response:**

```json
{
  "status": "success",
  "message": "Feedback submitted successfully for action: keep",
  "feedback_id": "feedback-uuid"
}
```

**Example cURL:**

```bash
curl -X POST "http://localhost:8000/api/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "query_id": "query-uuid",
    "result_id": "result-uuid",
    "action_type": "keep"
  }'
```

---

### 5. Vector Database Management Endpoints

#### POST `/api/vectordb/refresh`

**Description:** Manually refresh all VectorDB entries by regenerating embeddings

**Response:**

```json
{
  "message": "Successfully refreshed X VectorDB entries",
  "entries_updated": 5
}
```

#### GET `/api/vectordb/stats`

**Description:** Get statistics about the current VectorDB entries

**Response:**

```json
{
  "total_vector_entries": 10,
  "unique_industry_categories": 5,
  "industry_categories": ["FOOD", "ELECTRONICS", "AUTOMOTIVE"],
  "total_records_represented": 150,
  "average_quality_score": 0.85,
  "last_updated_entries": [
    {
      "industry": "FOOD",
      "last_updated": "2024-01-01T12:00:00"
    }
  ]
}
```

#### DELETE `/api/vectordb/industry/{industry_category}`

**Description:** Delete VectorDB entry for a specific industry category

**Path Parameters:**

- `industry_category` (string, required): Industry category to delete

**Response:**

```json
{
  "message": "Successfully deleted VectorDB entry for industry: FOOD",
  "deleted_industry": "FOOD"
}
```

**Error Responses:**

- `404`: Industry category not found

---

## Data Models

### SearchRequest

```json
{
  "query_text": "string",
  "filters": "string",
  "top_k": 5
}
```

### SearchResult

```json
{
  "company": "string",
  "product": "string",
  "completeness_score": 85,
  "semantic_score": 0.92,
  "doc_status": "string",
  "total_score": 88
}
```

### FeedbackRequest

```json
{
  "query_id": "string",
  "result_id": "string",
  "action_type": "string"
}
```

### FeedbackResponse

```json
{
  "status": "string",
  "message": "string",
  "feedback_id": "string"
}
```

---

## Error Handling

All endpoints return appropriate HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid parameters, missing required fields)
- `404`: Not Found (resource doesn't exist)
- `500`: Internal Server Error (server-side errors)

Error responses include a `detail` field with error description:

```json
{
  "detail": "Error message describing what went wrong"
}
```

---

## Environment Variables

Required environment variables:

- `OPENAI_API_KEY`: OpenAI API key for generating embeddings
- `DATABASE_URL`: Database connection string
- `DISABLE_DOCS`: Set to "true" to disable API documentation (optional)

---

## Installation and Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export DATABASE_URL="your-database-url"
```

3. Run the application:

```bash
python app.py
```

4. Access API documentation:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Features

### Semantic Search

- AI-powered semantic similarity search using OpenAI embeddings
- Natural language query processing
- Multi-factor scoring (completeness + semantic similarity)

### Data Processing

- Automatic file processing for Excel and CSV files
- Industry category grouping and scoring
- Data quality assessment with completeness scoring

### Vector Database

- PostgreSQL with pgvector extension for vector storage
- Automatic embedding generation and storage
- Vector similarity search capabilities

### Feedback System

- User feedback collection for search results
- Action tracking (keep, reject, compare)
- Query and result history storage

### Filtering

- Industry category filtering
- Country filtering (if available)
- Flexible filter format support

---

## API Documentation

Interactive API documentation is available at:

- Swagger UI: `/docs`
- ReDoc: `/redoc`

Both documentation interfaces provide:

- Interactive endpoint testing
- Request/response examples
- Schema definitions
- Parameter descriptions
