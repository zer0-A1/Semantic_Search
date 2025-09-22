# Semantic Search API

A FastAPI-based semantic search system for company and product data with AI-powered embeddings and vector similarity search.

## 🚀 Overview

This project provides a comprehensive semantic search API that processes company/product data from Excel/CSV files, generates embeddings using OpenAI's text-embedding-3-small model, and enables intelligent search through vector similarity matching. The system is designed for production deployment with PostgreSQL + pgvector for vector storage.

## 🏗️ Architecture

### Core Components

- **FastAPI Application**: Main web framework with async support
- **PostgreSQL + pgvector**: Vector database for embeddings storage
- **OpenAI API**: Text embedding generation
- **Pandas**: Data processing and analysis
- **SQLAlchemy**: ORM for database operations

### Database Schema

- **VectorDB**: Stores industry-categorized embeddings and metadata
- **SearchQuery**: Tracks search history and parameters
- **SearchResult**: Stores individual search results with scores
- **Feedback**: User feedback on search results

## 📁 Project Structure

```
semantic_search/
├── api/                    # API endpoints
│   ├── upload.py          # File upload and data processing
│   ├── search.py          # Semantic search functionality
│   └── feedback.py        # User feedback system
├── database/              # Database models and configuration
│   ├── database.py        # Database setup and models
│   └── schemas.py         # Pydantic schemas
├── app.py                 # Main FastAPI application
├── start.py               # Production startup script
├── requirements.txt       # Python dependencies
├── gunicorn.conf.py       # Gunicorn configuration
├── render.yaml           # Render.com deployment config
└── Procfile              # Process file for deployment
```

## 🔧 Installation & Setup

### Prerequisites

- Python 3.8+
- PostgreSQL with pgvector extension
- OpenAI API key

### Environment Variables

```bash
DATABASE_URL=postgresql://user:password@host:port/database
OPENAI_API_KEY=your_openai_api_key
DISABLE_DOCS=false  # Optional: disable API docs
```

### Installation

```bash
# Clone repository
git clone <repository-url>
cd semantic_search

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## 🚀 Deployment

### Render.com (Production)

The project is configured for Render.com deployment:

```bash
# Deploy using Render CLI or web interface
# Ensure DATABASE_URL and OPENAI_API_KEY are set in environment
```

### Local Development

```bash
# Development server
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Production server
python start.py
```

## 📚 API Endpoints

### Core Endpoints

#### Health Check

- **GET** `/` - Basic health check
- **GET** `/health` - Detailed health status

### Upload & Data Processing

#### Upload File

- **POST** `/api/upload`
- **Description**: Upload Excel/CSV files for data processing
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file`: Excel (.xlsx, .xls) or CSV file
- **Requirements**:
  - Must contain industry category column (e.g., '產業別', 'industry_category')
  - Data is grouped by industry and scored for quality
  - Recommended product/code and company columns (any that match):
    - Product: `問卷編號`, `產品編號`, `產品代號`, `product_id`, `product`, `sku`
    - Company: `客戶名稱`, `公司名稱`, `company`, `company_name`
- **Response**:

  ```json
  {
    "message": "Successfully processed X industry groups with embeddings",
    "file_id": "uuid",
    "groups_processed": 5,
    "data_type": "product"
  }
  ```

- **Metadata stored per industry group**:
  - `product_ids`: list of detected product IDs
  - `product_to_company`: mapping product_id → company
  - `product_metrics`: array of per-product objects:
    - `product_id`, `company`, `quantity`, `quality_score`, `tags`, `fields`
  - `numeric_fields`: list of detected numeric column names

#### VectorDB Management

##### Refresh VectorDB

- **POST** `/api/vectordb/refresh`
- **Description**: Regenerate all embeddings in VectorDB
- **Response**:
  ```json
  {
    "message": "Successfully refreshed X VectorDB entries",
    "entries_updated": 10
  }
  ```

##### Get VectorDB Statistics

- **GET** `/api/vectordb/stats`
- **Description**: Get comprehensive statistics about VectorDB
- **Response**:
  ```json
  {
    "total_vector_entries": 15,
    "unique_industry_categories": 8,
    "industry_categories": ["扣件", "食品", "電子"],
    "total_records_represented": 1500,
    "average_quality_score": 0.85,
    "last_updated_entries": [...]
  }
  ```

##### Delete Industry Entry

- **DELETE** `/api/vectordb/industry/{industry_category}`
- **Description**: Remove specific industry category from VectorDB
- **Response**:
  ```json
  {
    "message": "Successfully deleted VectorDB entry for industry: 扣件",
    "deleted_industry": "扣件"
  }
  ```

### Search Functionality

#### Semantic Search

- **POST** `/api/search`
- **Description**: Perform AI-powered semantic search with keyword extraction, product-code matching, and metric-aware selection
- **Request Body**:
  ```json
  {
    "query_text": "I need Q02 highest quantity product",
    "filters": "扣件", // or "industry:扣件,電子;country:TW"
    "top_k": 5
  }
  ```
- **Response**:

  ```json
  {
    "query_id": "uuid",
    "top_k": 5,
    "returned": 5,
    "results": [
      {
        "company": "ABC扣件公司",
        "product": "Q2024002",
        "completeness_score": 95,
        "semantic_score": 0.87,
        "doc_status": "有效",
        "total_score": 91
      }
    ]
  }
  ```

- **Query preprocessing features**:

  - Keyword extraction with basic stopwords
  - Product code detection with normalization (e.g., `Q02` ≈ `Q002`, hyphens ignored)
  - Simple synonyms (e.g., `two` → `2`)
  - Metric intent detection: “highest/lowest <metric>” maps to numeric fields (e.g., `quantity`, `quality_score`, or matching uploaded numeric columns via `fields`)

- **Ranking behavior**:
  - Score = 0.6 × completeness + 0.4 × semantic similarity
  - Boost vectors that contain an exact product code match
  - Honor metric intent by selecting the best product within top groups
  - Expand group results into product-level items up to `top_k`

#### Search History

- **GET** `/api/search/history?limit=10`
- **Description**: Get recent search queries
- **Response**:
  ```json
  {
    "queries": [
      {
        "id": "query-uuid",
        "query_text": "尋找高品質的扣件供應商",
        "filters": "扣件",
        "top_k": 5,
        "created_at": "2024-01-15T10:30:00"
      }
    ]
  }
  ```

#### Get Search Results

- **GET** `/api/search/results/{query_id}`
- **Description**: Retrieve results for specific search query
- **Response**:
  ```json
  {
    "query": {...},
    "results": [
      {
        "id": "result-uuid",
        "company": "ABC扣件公司",
        "product": "Q2024001",
        "completeness_score": 95,
        "semantic_score": 0.87,
        "doc_status": "有效",
        "total_score": 91,
        "rank": 1,
        "vector_id": "vector-uuid"
      }
    ]
  }
  ```

### Feedback System

#### Submit Feedback

- **POST** `/api/feedback`
- **Description**: Submit user feedback on search results
- **Request Body**:
  ```json
  {
    "query_id": "query-uuid",
    "result_id": "result-uuid",
    "action_type": "keep" // "keep", "reject", "compare"
  }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "message": "Feedback submitted successfully for action: keep",
    "feedback_id": "feedback-uuid"
  }
  ```

## 🔄 Data Workflow

### 1. Data Upload Process

1. **File Upload**: User uploads Excel/CSV file via `/api/upload`
2. **Data Validation**: System validates file format and required columns
3. **Industry Grouping**: Data is grouped by industry category (產業別)
4. **Quality Scoring**: Each record is scored based on:
   - Completeness (penalty for empty fields)
   - Date validity (expire_date, issue_date)
5. **Embedding Generation**: OpenAI API generates embeddings for each industry group
6. **Vector Storage**: Embeddings stored in PostgreSQL with pgvector

### 2. Search Process

1. **Query Processing**: User submits natural language query
2. **Filter Application**: System applies industry/country filters
3. **Embedding Generation**: Query converted to embedding vector
4. **Similarity Search**: Cosine similarity calculated against filtered vectors
5. **Multi-factor Scoring**: Results ranked by:
   - Completeness Score (60%)
   - Semantic Similarity (40%)
6. **Result Formatting**: Results formatted and returned to user

### 3. Quality Assurance

- **Document Status**: Automatic validation of expire dates
- **Score Calculation**: Comprehensive scoring system for data quality
- **Feedback Loop**: User feedback system for continuous improvement

## 🎯 Key Features

### Advanced Search Capabilities

- **Natural Language Processing**: Understands Chinese and English queries
- **Semantic Similarity**: AI-powered meaning-based search
- **Multi-factor Ranking**: Combines completeness and semantic scores
- **Flexible Filtering**: Industry and country-based filtering

### Data Quality Management

- **Automatic Scoring**: Quality assessment based on data completeness
- **Date Validation**: Expire date and issue date validation
- **Industry Categorization**: Automatic grouping by industry type

### Production Ready

- **Async Operations**: Full async/await support for high performance
- **Connection Pooling**: Optimized database connections
- **Error Handling**: Comprehensive error handling and logging
- **Scalable Architecture**: Designed for horizontal scaling

## 🔧 Configuration

### Database Configuration

- **PostgreSQL**: Primary database with pgvector extension
- **Connection Pooling**: 5 base connections, 10 max overflow
- **SSL Support**: Required for production deployments

### OpenAI Integration

- **Model**: text-embedding-3-small (1536 dimensions)
- **Rate Limiting**: Built-in error handling for API limits
- **Cost Optimization**: Efficient embedding generation

### Performance Tuning

- **Gunicorn Workers**: 4 workers for production
- **Timeout Settings**: 30-second request timeout
- **Memory Management**: Connection recycling and cleanup

## 📊 Monitoring & Analytics

### Built-in Statistics

- VectorDB entry counts and categories
- Search query history and patterns
- Data quality metrics and trends
- User feedback analysis

### Health Monitoring

- Database connection status
- API response times
- Error rates and logging

## 🛠️ Development

### Adding New Features

1. Create new router in `api/` directory
2. Add corresponding database models in `database/`
3. Update schemas in `database/schemas.py`
4. Include router in `app.py`

### Testing

```bash
# Run with test data
python -m pytest tests/

# Manual API testing
curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@test_data.xlsx"
```

## 📝 License

This project is proprietary software. All rights reserved.

## 🤝 Support

For technical support or questions, please contact the development team.

---

**Version**: 1.0.0  
**Last Updated**: 2024  
**Status**: Production Ready
