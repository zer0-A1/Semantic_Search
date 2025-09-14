from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, String, JSON, Float, ForeignKey, Boolean
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
    connect_args={"ssl": "require"},  # Enable SSL for the connection
    pool_size=5,  # Set a reasonable pool size
    max_overflow=10,  # Allow some overflow connections
    pool_timeout=30,  # Timeout for getting a connection from the pool
    pool_recycle=1800,  # Recycle connections after 30 minutes
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
    filter = Column(JSON)
    embedding = Column(Vector(1536))
    metadata_json = Column(JSON)
    
class SearchQuery(Base):
    __tablename__ = "SearchQuery"
    id = Column(String, primary_key=True, index=True)
    query = Column(String, nullable=False)
    top_k = Column(Integer, nullable=False)
    filter = Column(JSON)
    
class Result(Base):
    __tablename__ = "Result"
    id = Column(String, primary_key=True, index=True)
    query_id = Column(String, ForeignKey("SearchQuery.id"))
    result = Column(String, ForeignKey("VectorDB.id"))
    rank = Column(Integer, nullable=False)
    score = Column(Float, nullable=False)
    
class Feedback(Base):
    __tablename__ = "Feedback"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    query_id = Column(String, ForeignKey("SearchQuery.id"))
    result_id = Column(String, ForeignKey("Result.id"))
    
    action_type = Column(String, nullable=False)
    timestamp = Column(String, nullable=False)
    
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise  # Re-raise the exception so FastAPI can handle it