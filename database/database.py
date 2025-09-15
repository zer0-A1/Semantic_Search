from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, String, JSON, Float, ForeignKey, Boolean, text
from pgvector.sqlalchemy import Vector
import os
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs
from typing import AsyncGenerator
from sqlalchemy.orm import relationship

load_dotenv()

# Get database URL from environment variable and ensure proper formatting
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")

# Clean up any extra spaces in the URL
DATABASE_URL = DATABASE_URL.replace(" ", "")

# Parse the URL to handle SSL parameters
parsed = urlparse(DATABASE_URL)
query_params = parse_qs(parsed.query)

# Remove sslmode from query parameters as it's handled differently for asyncpg
if 'sslmode' in query_params:
    del query_params['sslmode']

# Reconstruct the URL without sslmode
clean_url = parsed._replace(query='').geturl()

# Ensure the URL starts with postgresql+asyncpg:// for async support
if not clean_url.startswith("postgresql+asyncpg://"):
    clean_url = clean_url.replace("postgresql://", "postgresql+asyncpg://")

# Create async engine with SSL configuration and connection pooling
engine = create_async_engine(
    clean_url,
    echo=True,
    connect_args={
        "ssl": "require",  # Enable SSL for the connection
        "prepared_statement_cache_size":
        0,  # Disable prepared statement caching
    },
    pool_size=5,  # Set a reasonable pool size
    max_overflow=10,  # Allow some overflow connections
    pool_timeout=30,  # Timeout for getting a connection from the pool
    pool_recycle=1800,  # Recycle connections after 30 minutes
    pool_pre_ping=True,  # Verify connections before use
)

# Create session factory with proper configuration
async_session = sessionmaker(engine,
                             class_=AsyncSession,
                             expire_on_commit=False,
                             autocommit=False,
                             autoflush=False)

Base = declarative_base()


class CompanyData(Base):
    __tablename__ = "CompanyData"

    id = Column(String, primary_key=True, index=True)
    CompanyName_CH = Column(String, nullable=False)
    CompanyName_EN = Column(String, nullable=False)
    Enterprise_Number = Column(String, nullable=False, unique=True)
    Industry_category = Column(String, nullable=False)
    Supplier_evaluation_information = Column(JSON)
    Phhone_Number = Column(String)
    Address = Column(String)
    Website = Column(String)
    Email = Column(String)
    Keywords = Column(JSON)
    Score = Column(Float)


class VectorDB(Base):
    __tablename__ = "VectorDB"
    id = Column(String, primary_key=True, index=True)
    filter = Column(String)
    embedding = Column(Vector(1536))
    metadata_json = Column(JSON)


class SearchQuery(Base):
    __tablename__ = "SearchQuery"
    id = Column(String, primary_key=True, index=True)
    query_text = Column(String, nullable=False)
    filters = Column(String)
    top_k = Column(Integer, nullable=False)
    created_at = Column(String,
                        nullable=False)  # Store as string for simplicity


class SearchResult(Base):
    __tablename__ = "SearchResult"
    id = Column(String, primary_key=True, index=True)
    query_id = Column(String, ForeignKey("SearchQuery.id"))
    company = Column(String, nullable=False)
    product = Column(String)
    completeness_score = Column(Integer, nullable=False)
    semantic_score = Column(Float, nullable=False)
    doc_status = Column(String, nullable=False)
    total_score = Column(Integer, nullable=False)
    rank = Column(Integer,
                  nullable=False)  # Position in results (1, 2, 3, etc.)
    vector_id = Column(
        String, ForeignKey("VectorDB.id"))  # Reference to original vector


class Feedback(Base):
    __tablename__ = "Feedback"
    id = Column(String, primary_key=True,
                index=True)  # Changed to String for UUID
    query_id = Column(String, ForeignKey("SearchQuery.id"))
    result_id = Column(String, ForeignKey("SearchResult.id"))

    action_type = Column(String, nullable=False)
    timestamp = Column(String, nullable=False)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Clear cached statements outside of transaction
    await clear_statement_cache()


async def clear_statement_cache():
    """Clear prepared statement cache to avoid cached statement errors."""
    try:
        # Invalidate all connections in the pool
        engine.dispose()

        # Wait a moment for connections to close
        import asyncio
        await asyncio.sleep(0.1)

        # Clear cache outside of transaction (DISCARD ALL cannot run in transaction)
        async with engine.connect() as conn:
            await conn.execute(text("DISCARD ALL"))
            await conn.commit()

    except Exception as e:
        print(f"Error clearing statement cache: {e}")
        # If clearing fails, at least dispose the engine to force new connections
        try:
            engine.dispose()
        except:
            pass


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise  # Re-raise the exception so FastAPI can handle it


async def save_search_query_and_results(query_text: str, filters: str,
                                        top_k: int, results: list,
                                        vector_ids: list) -> str:
    """Save search query and its results to the database."""
    import uuid
    from datetime import datetime

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
            for rank, (result,
                       vector_id) in enumerate(zip(results, vector_ids), 1):
                search_result = SearchResult(
                    id=str(uuid.uuid4()),
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

            await session.commit()
            return query_id

        except Exception as e:
            await session.rollback()
            raise e
