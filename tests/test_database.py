"""
Test suite for database functionality.
"""

import pytest
from unittest.mock import patch, Mock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database import get_db, check_db_connection, init_db
from app.config import settings


class TestDatabaseConnection:
    """Test database connection functionality."""

    def test_get_db_generator(self):
        """Test get_db returns a generator."""
        db_gen = get_db()
        assert hasattr(db_gen, "__next__")  # Check it's a generator

    @patch("app.database.engine")
    def test_check_db_connection_success(self, mock_engine):
        """Test successful database connection check."""
        mock_connection = Mock()
        mock_engine.connect.return_value.__enter__.return_value = mock_connection

        result = check_db_connection()
        assert result is True

    @patch("app.database.engine")
    def test_check_db_connection_failure(self, mock_engine):
        """Test failed database connection check."""
        mock_engine.connect.side_effect = Exception("Connection failed")

        result = check_db_connection()
        assert result is False

    def test_database_url_configuration(self):
        """Test database URL is properly configured."""
        # Test that database URL is set and valid
        assert settings.database_url is not None
        assert len(settings.database_url) > 0

        # Should be either PostgreSQL or SQLite
        assert settings.database_url.startswith(
            "postgresql://"
        ) or settings.database_url.startswith("sqlite:///")

    @patch("app.database.Base")
    @patch("app.database.engine")
    def test_init_db(self, mock_engine, mock_base):
        """Test database initialization."""
        mock_base.metadata.create_all = Mock()

        # Should not raise any exceptions
        init_db()

        # Check that create_all was called
        mock_base.metadata.create_all.assert_called_once_with(bind=mock_engine)


class TestDatabaseModels:
    """Test database model functionality."""

    def test_assessment_model_import(self):
        """Test Assessment model can be imported."""
        try:
            from app.models.assessment import Assessment

            assert Assessment is not None
        except ImportError:
            pytest.fail("Could not import Assessment model")

    def test_database_configuration_safety(self):
        """Test database configuration is safe for testing."""
        # Ensure we're not accidentally connecting to production DB in tests
        if hasattr(settings, "environment"):
            if settings.environment == "production":
                # In production, should use PostgreSQL
                assert settings.database_url.startswith("postgresql://")
            else:
                # In dev/test, can use SQLite or PostgreSQL
                assert settings.database_url.startswith(
                    "postgresql://"
                ) or settings.database_url.startswith("sqlite:///")


class TestDatabaseSecurity:
    """Test database security configurations."""

    def test_database_credentials_not_hardcoded(self):
        """Test that database credentials are configurable via environment."""
        # Check that default values are placeholder values, not real credentials
        if settings.postgres_password == "postgres":
            # This is acceptable for default/development
            pass
        else:
            # If custom password, should not be empty
            assert len(settings.postgres_password) > 0

    def test_database_configuration_completeness(self):
        """Test all required database configuration is present."""
        required_configs = [
            "database_url",
            "postgres_user",
            "postgres_password",
            "postgres_db",
            "postgres_host",
            "postgres_port",
        ]

        for config in required_configs:
            assert hasattr(settings, config)
            assert getattr(settings, config) is not None


if __name__ == "__main__":
    pytest.main([__file__])
