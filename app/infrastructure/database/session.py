from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.infrastructure.database.models import Base
from app.utils.logger import get_logger

logger = get_logger(__name__)

# SQLite database (simple, no installation needed)
DATABASE_URL = "sqlite:///./rag_system.db"

# Create engine
engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False}  # Needed for SQLite
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initialize database (create tables)"""
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created")

def get_db():
    """Get database session (for FastAPI dependency injection)"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()