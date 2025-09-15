# Semantic Search API

A powerful FastAPI-based semantic search application that enables intelligent search across company and product data using OpenAI embeddings and vector similarity search. The system supports natural language queries, advanced filtering, and user feedback collection.

## 🚀 Features

### Core Functionality

- **Semantic Search**: Natural language search using OpenAI embeddings
- **Multi-format Data Upload**: Support for Excel (.xlsx, .xls) and CSV files
- **Advanced Filtering**: Industry category and custom filter support
- **Vector Database**: PostgreSQL with pgvector extension for efficient similarity search
- **User Feedback System**: Collect and analyze user interactions with search results
- **Search History**: Track and retrieve previous search queries
- **Scoring System**: Multi-factor scoring based on completeness, semantic similarity, and data quality

### Data Types Supported

- **Company Data**: Company profiles with evaluation information
- **Product Data**: Product information grouped by industry categories

## 🏗️ Architecture

### Technology Stack

- **Backend**: FastAPI (Python 3.11+)
- **Database**: PostgreSQL with pgvector extension
- **AI/ML**: OpenAI API for embeddings
- **Data Processing**: Pandas for data manipulation
- **ORM**: SQLAlchemy with async support
- **Vector Search**: Cosine similarity with pgvector

### Project Structure

```
semantic_search/
├── api/                    # API endpoints
│   ├── upload.py          # File upload and data processing
│   ├── search.py          # Semantic search functionality
│   └── feedback.py        # User feedback system
├── database/              # Database models and schemas
│   ├── database.py        # SQLAlchemy models and database setup
│   └── schemas.py         # Pydantic models for API validation
├── app.py                 # FastAPI application entry point
├── .env                   # Environment variables
└── README.md             # This file
```

## 📋 Prerequisites

- Python 3.11 or higher
- PostgreSQL 12+ with pgvector extension
- OpenAI API key
- pip or conda package manager

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd semantic_search
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n semantic_search python=3.11
conda activate semantic_search
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Database Setup

1. Install PostgreSQL with pgvector extension
2. Create a database for the application
3. Set up the connection string in your environment

### 5. Environment Configuration

Create a `.env` file in the project root:

```env
DATABASE_URL=postgresql://username:password@localhost:5432/database_name
OPENAI_API_KEY=your_openai_api_key_here
```

### 6. Initialize Database

```bash
python -c "from database.database import init_db; import asyncio; asyncio.run(init_db())"
```

## 🚀 Running the Application

### Development Mode

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## 📚 API Documentation

### Interactive Documentation

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### API Endpoints

#### Upload Endpoints

- **POST** `/api/upload` - Upload Excel/CSV files for processing

#### Search Endpoints

- **POST** `/api/search` - Perform semantic search
- **GET** `/api/search/history` - Get search history
- **GET** `/api/search/results/{query_id}` - Get results for specific query

#### Feedback Endpoints

- **POST** `/api/feedback` - Submit user feedback
- **GET** `/api/feedback/query/{query_id}` - Get feedback for specific query
- **GET** `/api/feedback/stats` - Get feedback statistics
- **DELETE** `/api/feedback/{feedback_id}` - Delete specific feedback

## 📖 Usage Examples

### 1. Upload Company Data

```bash
curl -X POST "http://localhost:8000/api/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@company_data.xlsx"
```

### 2. Perform Semantic Search

```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "尋找高品質的扣件供應商",
    "filters": "扣件",
    "top_k": 5
  }'
```

### 3. Submit Feedback

```bash
curl -X POST "http://localhost:8000/api/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "query_id": "query-uuid-here",
    "result_id": "result-uuid-here",
    "action_type": "keep"
  }'
```

## 🔍 Search Features

### Filter Formats

The search supports two filter formats:

1. **Simple Format**: `"扣件"` → Filters by industry category
2. **Complex Format**: `"industry:FOOD,FOOD2;country:VN,TH"` → Multiple filter criteria

### Scoring System

Results are scored based on:

- **Completeness Score (60%)**: Data completeness and quality
- **Semantic Similarity (40%)**: Vector similarity to query
- **Document Status**: Current status of company documents

### Search Capabilities

- Natural language queries in multiple languages
- Fuzzy matching and tolerance for typos
- Multi-factor sorting and ranking
- Industry-specific filtering
- Historical search tracking

## 📊 Data Models

### Company Data Schema

```python
{
    "CompanyName_CH": str,           # Chinese company name
    "CompanyName_EN": str,           # English company name
    "Enterprise_Number": str,        # Unique enterprise number
    "Industry_category": str,        # Industry classification
    "Supplier_evaluation_information": dict,  # Evaluation data
    "Phone_Number": str,             # Contact phone
    "Address": str,                  # Company address
    "Website": str,                  # Company website
    "Email": str,                    # Contact email
    "Keywords": list,                # Associated keywords
    "Score": float                   # Quality score
}
```

### Search Request Schema

```python
{
    "query_text": str,               # Search query
    "filters": str,                  # Filter criteria
    "top_k": int                     # Number of results to return
}
```

### Search Result Schema

```python
{
    "company": str,                  # Company name
    "product": str,                  # Product information
    "completeness_score": int,       # Data completeness (0-100)
    "semantic_score": float,         # Semantic similarity (0-1)
    "doc_status": str,               # Document status
    "total_score": int               # Overall score (0-100)
}
```

## 🔧 Configuration

### Database Configuration

The application uses PostgreSQL with the following key settings:

- Connection pooling with 5 base connections
- SSL required for secure connections
- Prepared statement caching disabled for compatibility
- Connection recycling every 30 minutes

### OpenAI Configuration

- Uses OpenAI's text-embedding-ada-002 model
- 1536-dimensional embeddings
- Automatic retry logic for API calls

## 🐛 Troubleshooting

### Common Issues

1. **Database Connection Errors**

   - Verify PostgreSQL is running
   - Check DATABASE_URL format
   - Ensure pgvector extension is installed

2. **OpenAI API Errors**

   - Verify OPENAI_API_KEY is set
   - Check API key permissions
   - Monitor API rate limits

3. **File Upload Issues**
   - Ensure file format is supported (.csv, .xlsx, .xls)
   - Check file size limits
   - Verify required columns are present

### Debug Mode

Enable debug logging by setting the log level in your environment:

```env
LOG_LEVEL=DEBUG
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:

- Create an issue in the repository
- Check the API documentation at `/docs`
- Review the troubleshooting section above

## 🔄 Version History

- **v1.0.0**: Initial release with basic semantic search
- **v1.1.0**: Added feedback system and search history
- **v1.2.0**: Enhanced filtering and scoring algorithms

---

**Note**: This application requires an active internet connection for OpenAI API calls and a properly configured PostgreSQL database with pgvector extension.
