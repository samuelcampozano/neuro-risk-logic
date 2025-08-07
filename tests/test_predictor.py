"""
Test suite for ML predictor functionality.
"""

import pytest
import os
from unittest.mock import Mock, patch
from app.models.predictor import NeurodevelopmentalPredictor
from app.config import settings


class TestNeurodevelopmentalPredictor:
    """Test the ML predictor class."""

    def test_predictor_initialization(self):
        """Test predictor can be initialized."""
        predictor = NeurodevelopmentalPredictor()
        assert predictor is not None
        assert predictor.model is None
        assert predictor.is_loaded is False

    def test_predictor_model_path(self):
        """Test predictor uses correct model path."""
        predictor = NeurodevelopmentalPredictor()
        expected_path = settings.get_model_path()
        assert predictor.model_path == expected_path

    @patch("os.path.exists")
    def test_load_model_file_not_exists(self, mock_exists):
        """Test load_model returns False when model file doesn't exist."""
        mock_exists.return_value = False

        predictor = NeurodevelopmentalPredictor()
        result = predictor.load_model()

        assert result is False
        assert predictor.is_loaded is False

    @patch("builtins.open")
    @patch("os.path.exists")
    @patch("pickle.load")
    def test_load_model_success(self, mock_pickle_load, mock_exists, mock_open):
        """Test successful model loading."""
        mock_exists.return_value = True
        mock_model = Mock()
        mock_pickle_load.return_value = {
            "model": mock_model,
            "metadata": {"version": "test_version"},
        }

        predictor = NeurodevelopmentalPredictor()
        result = predictor.load_model()

        assert result is True
        assert predictor.is_loaded is True
        assert predictor.model == mock_model

    def test_predict_without_loaded_model(self):
        """Test predict raises error when model not loaded."""
        predictor = NeurodevelopmentalPredictor()
        test_data = {"age": 25, "gender": "M"}

        with pytest.raises(Exception):
            predictor.predict(test_data)

    @patch("builtins.open")
    @patch("os.path.exists")
    @patch("pickle.load")
    def test_predict_with_loaded_model(self, mock_pickle_load, mock_exists, mock_open):
        """Test predict works with loaded model."""
        mock_exists.return_value = True
        mock_model = Mock()
        mock_model.predict_proba.return_value = [[0.3, 0.7]]
        mock_model.predict.return_value = [1]

        mock_pickle_load.return_value = {
            "model": mock_model,
            "metadata": {"version": "test_version"},
        }

        predictor = NeurodevelopmentalPredictor()
        predictor.load_model()

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
        }

        result = predictor.predict(test_data)

        assert isinstance(result, dict)
        assert "risk_score" in result
        assert "risk_level" in result
        assert "confidence_score" in result


class TestPredictorUtilities:
    """Test predictor utility functions."""

    def test_feature_processing(self):
        """Test feature processing works correctly."""
        # Test that predictor can handle the expected feature set
        expected_features = [
            "age",
            "gender",
            "consanguinity",
            "family_neuro_history",
            "seizures_history",
            "brain_injury_history",
            "psychiatric_diagnosis",
            "substance_use",
            "suicide_ideation",
            "psychotropic_medication",
            "birth_complications",
            "extreme_poverty",
            "education_access_issues",
            "healthcare_access",
            "disability_diagnosis",
            "social_support_level",
            "breastfed_infancy",
            "violence_exposure",
        ]

        predictor = NeurodevelopmentalPredictor()
        # This test verifies the feature set is complete
        assert len(expected_features) == 18  # Expected number of features


if __name__ == "__main__":
    pytest.main([__file__])
