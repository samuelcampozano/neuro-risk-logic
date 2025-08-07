"""
Database configuration and session management.
Supports both PostgreSQL (production) and SQLite (development).
"""

import os
from typing import Generator
from sqlalchemy import create_engine, event, MetaData, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool, NullPool

from app.config import settings
from loguru import logger

# Determine engine configuration based on database type
if settings.database_url.startswith("sqlite"):
    # SQLite configuration for development
    engine = create_engine(
        settings.database_url,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=settings.debug,
    )

    # Enable foreign keys for SQLite
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    logger.info("Using SQLite database for development")

elif settings.database_url.startswith("postgresql"):
    # PostgreSQL configuration for production
    engine = create_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_recycle=300,
        pool_size=10,
        max_overflow=20,
        echo=settings.debug,
    )
    logger.info("PostgreSQL engine created (connection will be established on first use)")

else:
    raise ValueError(f"Unsupported database URL: {settings.database_url}")

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class for models
Base = declarative_base()

# Metadata for migrations
metadata = MetaData()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency function to get database session.
    Ensures proper cleanup after request completion.

    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Initialize database with all tables.
    Creates tables if they don't exist.
    """
    # Import all models to ensure they're registered
    from app.models import assessment  # noqa

    # Create data directory for SQLite if needed
    if settings.database_url.startswith("sqlite"):
        os.makedirs("data", exist_ok=True)

    # Create all tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")


def check_db_connection() -> bool:
    """
    Test database connection.

    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        with engine.connect() as conn:
            # Use text() for raw SQL
            conn.execute(text("SELECT 1"))
        logger.info("Database connection check successful")
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {str(e)}")
        return False


def ensure_postgresql_connection(max_retries: int = 5, retry_interval: int = 5) -> bool:
    """
    Ensure PostgreSQL connection with retry logic.
    Only needed for production environments.

    Args:
        max_retries: Maximum number of connection attempts
        retry_interval: Seconds to wait between retries

    Returns:
        bool: True if connection successful, False otherwise
    """
    if not settings.database_url.startswith("postgresql"):
        return True  # Not PostgreSQL, no need to retry

    import time
    from sqlalchemy.exc import OperationalError

    for attempt in range(max_retries):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Successfully connected to PostgreSQL database")
            return True
        except OperationalError as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"Failed to connect to PostgreSQL "
                    f"(attempt {attempt + 1}/{max_retries}): {str(e)}"
                )
                logger.info(f"Retrying in {retry_interval} seconds...")
                time.sleep(retry_interval)
            else:
                logger.error(f"Failed to connect to PostgreSQL after {max_retries} attempts")
                return False
    return False


def get_db_stats() -> dict:
    """
    Get database statistics and information.

    Returns:
        dict: Database statistics
    """
    from app.models.assessment import Assessment

    db = SessionLocal()
    try:
        stats = {
            "engine": engine.url.drivername,
            "database": engine.url.database,
            "tables": list(Base.metadata.tables.keys()),
            "assessment_count": db.query(Assessment).count(),
            "connection_pool": {
                "size": getattr(engine.pool, "size", "N/A"),
                "checked_out": getattr(engine.pool, "checked_out", "N/A"),
            },
        }
        return stats
    except Exception as e:
        logger.error(f"Error getting database stats: {str(e)}")
        return {"error": str(e)}
    finally:
        db.close()


# Optional: Alembic integration for migrations
def create_migration_engine():
    """
    Create engine specifically for Alembic migrations.
    """
    return engine
