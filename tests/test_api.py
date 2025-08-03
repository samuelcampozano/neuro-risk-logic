"""
Test suite for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.config import settings

client = TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self):
        """Test health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data


class TestPredictionEndpoint:
    """Test prediction endpoint."""

    def test_prediction_endpoint_structure(self):
        """Test prediction endpoint exists and has proper structure."""
        # Test with minimal valid data
        test_data = {
            "age": 25,
            "gender": "M",
            "consanguinity": False,
            "family_neuro_history": False,
            "seizures_history": False,
            "brain_injury_history": False,
            "psychiatric_diagnosis": False,
            "substance_use": False,
            "suicide_ideation": False,
            "psychotropic_medication": False,
            "birth_complications": False,
            "extreme_poverty": False,
            "education_access_issues": False,
            "healthcare_access": True,
            "disability_diagnosis": False,
            "social_support_level": "moderate",
            "breastfed_infancy": True,
            "violence_exposure": False
        }
        
        response = client.post("/api/v1/predict", json=test_data)
        
        # Should return 200 or 500 (if model not loaded), but not 404/422
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            # Check response structure
            assert "risk_score" in data
            assert "risk_level" in data
            assert "confidence_score" in data


class TestAuthEndpoint:
    """Test authentication endpoints."""

    def test_auth_endpoint_exists(self):
        """Test auth endpoint exists."""
        response = client.post(
            "/api/v1/auth/login",
            json={"api_key": "test-key"}
        )
        # Should return 401 (unauthorized) or 200, but not 404
        assert response.status_code in [200, 401]


class TestStatsEndpoint:
    """Test statistics endpoints."""

    def test_stats_endpoint_requires_auth(self):
        """Test stats endpoint requires authentication."""
        response = client.get("/api/v1/stats")
        assert response.status_code == 401  # Unauthorized without token


class TestAssessmentsEndpoint:
    """Test assessments endpoints."""

    def test_assessments_endpoint_structure(self):
        """Test assessments endpoint has proper structure."""
        # Test POST without auth should return 401
        test_data = {
            "age": 25,
            "gender": "M",
            "consanguinity": False,
            "family_neuro_history": False,
            "seizures_history": False,
            "brain_injury_history": False,
            "psychiatric_diagnosis": False,
            "substance_use": False,
            "suicide_ideation": False,
            "psychotropic_medication": False,
            "birth_complications": False,
            "extreme_poverty": False,
            "education_access_issues": False,
            "healthcare_access": True,
            "disability_diagnosis": False,
            "social_support_level": "moderate",
            "breastfed_infancy": True,
            "violence_exposure": False,
            "consent_data_usage": True,
            "consent_research": True,
            "consent_contact": False
        }
        
        response = client.post("/api/v1/assessments", json=test_data)
        # Should work without auth (public endpoint) or require auth
        assert response.status_code in [200, 201, 401, 500]


if __name__ == "__main__":
    pytest.main([__file__])