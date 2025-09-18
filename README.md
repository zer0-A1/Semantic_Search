# 🔍 Semantic Search API

A powerful FastAPI-based semantic search application that enables intelligent search across company and product data using OpenAI embeddings and vector similarity search. The system supports natural language queries, advanced filtering, and user feedback collection with production-ready deployment on Render.com.

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)](https://postgresql.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/)
[![Render](https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com/)

## 🌟 Live Demo

**🚀 Production API**: [https://semantic-search-vfez.onrender.com/](https://semantic-search-vfez.onrender.com/)

- **📚 API Documentation**: [https://semantic-search-vfez.onrender.com/docs](https://semantic-search-vfez.onrender.com/docs)
- **📖 ReDoc Documentation**: [https://semantic-search-vfez.onrender.com/redoc](https://semantic-search-vfez.onrender.com/redoc)
- **❤️ Health Check**: [https://semantic-search-vfez.onrender.com/health](https://semantic-search-vfez.onrender.com/health)

## ✨ Features

### 🔍 Core Functionality

- **🧠 Semantic Search**: Natural language search using OpenAI embeddings
- **📊 Multi-format Data Upload**: Support for Excel (.xlsx, .xls) and CSV files
- **🎯 Advanced Filtering**: Industry category and custom filter support
- **🗄️ Vector Database**: PostgreSQL with pgvector extension for efficient similarity search
- **💬 User Feedback System**: Collect and analyze user interactions with search results
- **📈 Search History**: Track and retrieve previous search queries
- **⭐ Scoring System**: Multi-factor scoring based on completeness, semantic similarity, and data quality

### 🚀 Production Features

- **☁️ Cloud Deployment**: Ready for Render.com deployment
- **🔄 Auto-scaling**: Gunicorn with multiple workers
- **📊 Monitoring**: Health checks and logging
- **🔒 Security**: SSL/TLS encryption and secure database connections
- **📱 API Documentation**: Interactive Swagger UI and ReDoc

### 📁 Data Types Supported

- **🏢 Company Data**: Company profiles with evaluation information
- **📦 Product Data**: Product information grouped by industry categories

## 🏗️ Architecture

### 🛠️ Technology Stack

- **Backend**: FastAPI (Python 3.11+)
- **Database**: PostgreSQL with pgvector extension
- **AI/ML**: OpenAI API for embeddings
- **Data Processing**: Pandas for data manipulation
- **ORM**: SQLAlchemy with async support
- **Vector Search**: Cosine similarity with pgvector
- **Production Server**: Gunicorn with Uvicorn workers
- **Deployment**: Render.com

### 📂 Project Structure

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
├── start.py               # Production startup script
├── gunicorn.conf.py       # Gunicorn configuration
├── Procfile               # Render.com deployment configuration
├── render.yaml            # Render.com service configuration
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (local)
└── README.md             # This file
```

## 🚀 Quick Start

### 🐳 One-Click Deploy to Render.com

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

1. Click the "Deploy to Render" button above
2. Connect your GitHub repository
3. Set environment variables:
   - `DATABASE_URL`: Your PostgreSQL connection string
   - `OPENAI_API_KEY`: Your OpenAI API key
4. Deploy! 🎉

### 🏠 Local Development

#### Prerequisites

- Python 3.11 or higher
- PostgreSQL 12+ with pgvector extension
- OpenAI API key

#### Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd semantic_search

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# 5. Initialize database
python -c "from database.database import init_db; import asyncio; asyncio.run(init_db())"

# 6. Run the application
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## 📚 API Documentation

### 🔗 Endpoints Overview

| Method   | Endpoint                         | Description                   |
| -------- | -------------------------------- | ----------------------------- |
| `GET`    | `/`                              | Root endpoint with API status |
| `GET`    | `/health`                        | Detailed health check         |
| `GET`    | `/docs`                          | Interactive Swagger UI        |
| `GET`    | `/redoc`                         | ReDoc documentation           |
| `POST`   | `/api/upload`                    | Upload Excel/CSV files        |
| `POST`   | `/api/search`                    | Perform semantic search       |
| `GET`    | `/api/search/history`            | Get search history            |
| `GET`    | `/api/search/results/{query_id}` | Get specific query results    |
| `POST`   | `/api/feedback`                  | Submit user feedback          |
| `GET`    | `/api/feedback/query/{query_id}` | Get feedback for query        |
| `GET`    | `/api/feedback/stats`            | Get feedback statistics       |
| `DELETE` | `/api/feedback/{feedback_id}`    | Delete feedback               |

### 📖 Usage Examples

#### 1. Upload Company Data

```bash
curl -X POST "https://semantic-search-vfez.onrender.com/api/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@company_data.xlsx"
```

#### 2. Perform Semantic Search

```bash
curl -X POST "https://semantic-search-vfez.onrender.com/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "尋找高品質的扣件供應商",
    "filters": "扣件",
    "top_k": 5
  }'
```

#### 3. Submit Feedback

```bash
curl -X POST "https://semantic-search-vfez.onrender.com/api/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "query_id": "query-uuid-here",
    "result_id": "result-uuid-here",
    "action_type": "keep"
  }'
```

## 🔍 Search Features

### 🎯 Filter Formats

The search supports two filter formats:

1. **Simple Format**: `"扣件"` → Filters by industry category
2. **Complex Format**: `"industry:FOOD,FOOD2;country:VN,TH"` → Multiple filter criteria

### ⭐ Scoring System

Results are scored based on:

- **Completeness Score (60%)**: Data completeness and quality
- **Semantic Similarity (40%)**: Vector similarity to query
- **Document Status**: Current status of company documents

### 🔍 Search Capabilities

- Natural language queries in multiple languages
- Fuzzy matching and tolerance for typos
- Multi-factor sorting and ranking
- Industry-specific filtering
- Historical search tracking

## 📊 Data Models

### 🏢 Company Data Schema

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

### 🔍 Search Request Schema

```python
{
    "query_text": str,               # Search query
    "filters": str,                  # Filter criteria
    "top_k": int                     # Number of results to return
}
```

### 📈 Search Result Schema

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

## ⚙️ Configuration

### 🗄️ Database Configuration

The application uses PostgreSQL with optimized settings:

- **Connection Pooling**: 5 base connections with 10 max overflow
- **SSL Required**: Secure connections for production
- **Connection Recycling**: Every 30 minutes to prevent memory leaks
- **Pre-ping**: Verifies connections before use
- **Statement Caching**: Optimized for performance

### 🤖 OpenAI Configuration

- **Model**: text-embedding-ada-002
- **Dimensions**: 1536-dimensional embeddings
- **Retry Logic**: Automatic retry for API calls
- **Rate Limiting**: Built-in rate limit handling

### 🚀 Production Configuration

- **Server**: Gunicorn with Uvicorn workers
- **Workers**: 4 workers (configurable via `WEB_CONCURRENCY`)
- **Timeout**: 30 seconds for request handling
- **Memory Management**: Worker recycling after 1000 requests
- **Logging**: Structured logging for monitoring

## 🐛 Troubleshooting

### ❌ Common Issues

#### 1. Database Connection Errors

```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Verify connection string format
echo $DATABASE_URL
# Should be: postgresql://user:password@host:port/database
```

#### 2. OpenAI API Errors

```bash
# Verify API key
echo $OPENAI_API_KEY

# Check API key permissions
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models
```

#### 3. File Upload Issues

- Ensure file format is supported (.csv, .xlsx, .xls)
- Check file size limits (default: 10MB)
- Verify required columns are present

#### 4. Render.com Deployment Issues

- Check build logs in Render.com dashboard
- Verify environment variables are set
- Ensure `Procfile` is in root directory
- Check port binding configuration

### 🔧 Debug Mode

Enable debug logging:

```env
LOG_LEVEL=DEBUG
DISABLE_DOCS=false
```

### 📊 Health Monitoring

Check application health:

```bash
# Basic health check
curl https://semantic-search-vfez.onrender.com/health

# Detailed status
curl https://semantic-search-vfez.onrender.com/
```

## 🚀 Deployment

### 🌐 Render.com Deployment

1. **Connect Repository**: Link your GitHub repository
2. **Set Environment Variables**:
   - `DATABASE_URL`: PostgreSQL connection string
   - `OPENAI_API_KEY`: OpenAI API key
   - `WEB_CONCURRENCY`: Number of workers (optional)
   - `ENVIRONMENT`: production (optional)
3. **Deploy**: Render.com will automatically build and deploy

### 🐳 Docker Deployment (Optional)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker"]
```

### 🔧 Environment Variables

| Variable          | Description                          | Required | Default     |
| ----------------- | ------------------------------------ | -------- | ----------- |
| `DATABASE_URL`    | PostgreSQL connection string         | ✅       | -           |
| `OPENAI_API_KEY`  | OpenAI API key                       | ✅       | -           |
| `WEB_CONCURRENCY` | Number of Gunicorn workers           | ❌       | 4           |
| `ENVIRONMENT`     | Environment (production/development) | ❌       | development |
| `DISABLE_DOCS`    | Disable API documentation            | ❌       | false       |
| `LOG_LEVEL`       | Logging level                        | ❌       | INFO        |

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### 🧪 Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run linting
flake8 .

# Format code
black .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### 📞 Getting Help

- **📚 Documentation**: [https://semantic-search-vfez.onrender.com/docs](https://semantic-search-vfez.onrender.com/docs)
- **🐛 Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)

### 🔗 Useful Links

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [Render.com Documentation](https://render.com/docs)

## 🔄 Version History

- **v1.0.0**: Initial release with basic semantic search
- **v1.1.0**: Added feedback system and search history
- **v1.2.0**: Enhanced filtering and scoring algorithms
- **v1.3.0**: Production deployment with Gunicorn and Render.com
- **v1.4.0**: Improved error handling and database optimization

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) for the amazing web framework
- [OpenAI](https://openai.com/) for the embedding models
- [PostgreSQL](https://postgresql.org/) and [pgvector](https://github.com/pgvector/pgvector) for vector search
- [Render.com](https://render.com/) for hosting and deployment

---

**⚡ Ready to search semantically?** [Try the live API](https://semantic-search-vfez.onrender.com/docs) or [deploy your own](https://render.com/deploy)!

**📝 Note**: This application requires an active internet connection for OpenAI API calls and a properly configured PostgreSQL database with pgvector extension.
