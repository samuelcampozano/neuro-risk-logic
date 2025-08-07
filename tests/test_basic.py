"""
Basic test suite to ensure CI/CD pipeline works.
These tests don't require complex dependencies or model loading.
"""

import pytest
import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestProjectStructure:
    """Test basic project structure and imports."""

    def test_project_root_exists(self):
        """Test project root directory exists."""
        assert project_root.exists()
        assert project_root.is_dir()

    def test_app_directory_exists(self):
        """Test app directory exists."""
        app_dir = project_root / "app"
        assert app_dir.exists()
        assert app_dir.is_dir()

    def test_config_can_be_imported(self):
        """Test config module can be imported."""
        try:
            from app.config import settings

            assert settings is not None
            assert hasattr(settings, "app_name")
        except ImportError as e:
            pytest.fail(f"Failed to import config: {e}")

    def test_basic_settings(self):
        """Test basic settings are properly configured."""
        from app.config import settings

        # Test required settings exist
        assert hasattr(settings, "app_name")
        assert hasattr(settings, "database_url")
        assert hasattr(settings, "secret_key")

        # Test app name is set
        assert settings.app_name == "NeuroRiskLogic"


class TestUtilities:
    """Test utility functions."""

    def test_feature_definitions_module_exists(self):
        """Test feature definitions module exists."""
        feature_def_file = project_root / "app" / "utils" / "feature_definitions.py"
        assert feature_def_file.exists()

    def test_risk_calculator_module_exists(self):
        """Test risk calculator module exists."""
        risk_calc_file = project_root / "app" / "utils" / "risk_calculator.py"
        assert risk_calc_file.exists()


class TestDataDirectories:
    """Test data directory structure."""

    def test_data_directory_exists(self):
        """Test data directory exists."""
        data_dir = project_root / "data"
        assert data_dir.exists()
        assert data_dir.is_dir()

    def test_models_directory_exists(self):
        """Test models directory exists."""
        models_dir = project_root / "data" / "models"
        assert models_dir.exists()
        assert models_dir.is_dir()

    def test_feature_definitions_file_exists(self):
        """Test feature definitions file exists."""
        feature_def_file = project_root / "data" / "feature_definitions.json"
        assert feature_def_file.exists()


class TestSecurityBasics:
    """Test basic security configurations."""

    def test_gitignore_exists(self):
        """Test .gitignore exists and contains sensitive patterns."""
        gitignore_file = project_root / ".gitignore"
        assert gitignore_file.exists()

        with open(gitignore_file, "r") as f:
            content = f.read()
            # Check for important patterns
            assert ".env" in content
            assert "*.key" in content
            assert "*.pem" in content

    def test_env_example_exists(self):
        """Test .env.example exists."""
        env_example_file = project_root / ".env.example"
        assert env_example_file.exists()


class TestRequirements:
    """Test requirements and dependencies."""

    def test_requirements_file_exists(self):
        """Test requirements.txt exists."""
        requirements_file = project_root / "requirements.txt"
        assert requirements_file.exists()

    def test_pytest_available(self):
        """Test pytest is available."""
        try:
            import pytest

            assert pytest is not None
        except ImportError:
            pytest.fail("pytest not available")


class TestDocumentation:
    """Test documentation files."""

    def test_readme_exists(self):
        """Test README.md exists."""
        readme_file = project_root / "README.md"
        assert readme_file.exists()

    def test_license_exists(self):
        """Test LICENSE exists."""
        license_file = project_root / "LICENSE"
        assert license_file.exists()


if __name__ == "__main__":
    pytest.main([__file__])
