"""
Machine Learning predictor for mental health risk assessment.
"""

import os
import pickle
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from datetime import datetime
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from loguru import logger

from app.config import settings
from app.utils.feature_definitions import load_feature_definitions
from app.utils.risk_calculator import (
    calculate_risk_factors,
    generate_recommendations,
    interpret_risk_level,
)


class NeurodevelopmentalPredictor:
    """Handles ML model loading and predictions."""

    def __init__(self):
        """Initialize predictor with model and feature definitions."""
        self.model: Optional[RandomForestClassifier] = None
        self.feature_definitions = load_feature_definitions()
        self.model_path = settings.get_model_path()
        self.model_metadata: Dict[str, Any] = {}
        self.is_loaded = False

    def load_model(self) -> bool:
        """
        Load the trained model from disk.

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.model_path.exists():
                logger.error(f"Model file not found at {self.model_path}")
                return False

            with open(self.model_path, "rb") as f:
                model_data = pickle.load(f)

            self.model = model_data["model"]
            self.model_metadata = model_data.get("metadata", {})
            self.is_loaded = True

            model_version = self.model_metadata.get("version", "unknown")
            logger.info(f"Model loaded successfully. Version: {model_version}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def predict(self, assessment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a risk prediction based on assessment data.

        Args:
            assessment_data: Dictionary with assessment features

        Returns:
            Dictionary with prediction results
        """
        if not self.is_loaded and not self.load_model():
            raise RuntimeError("Model not loaded")

        try:
            # Convert to feature vector
            feature_vector = self.feature_definitions.get_feature_vector(assessment_data)
            feature_array = np.array([feature_vector])

            # Make prediction
            risk_probability = self.model.predict_proba(feature_array)[0, 1]
            risk_prediction = self.model.predict(feature_array)[0]

            # Calculate confidence (based on prediction probability)
            confidence = max(self.model.predict_proba(feature_array)[0])

            # Get feature importances if available
            feature_importances = {}
            if hasattr(self.model, "feature_importances_"):
                feature_names = self.feature_definitions.get_feature_names()
                for idx, importance in enumerate(self.model.feature_importances_):
                    if idx < len(feature_names):
                        feature_importances[feature_names[idx]] = float(importance)

            # Calculate risk factors and protective factors
            risk_factors, protective_factors = calculate_risk_factors(
                assessment_data, feature_importances
            )

            # Ensure protective factors always has at least one message
            if not protective_factors:
                protective_factors = ["No protective factors were identified"]

            # Get risk level
            risk_level = self.feature_definitions.get_risk_level(risk_probability)

            # Generate recommendations
            recommendations = generate_recommendations(risk_level, risk_factors, assessment_data)

            # Get interpretation
            interpretation = interpret_risk_level(risk_probability, confidence)

            # Compile results
            results = {
                "risk_score": float(risk_probability),
                "risk_level": risk_level,
                "confidence_score": float(confidence),
                "risk_factors": risk_factors,
                "protective_factors": protective_factors,
                "feature_importance": feature_importances,
                "recommendations": recommendations,
                "interpretation": interpretation,
                "model_version": self.model_metadata.get("version", settings.model_version),
                "prediction_timestamp": datetime.utcnow().isoformat(),
            }

            return results

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model metadata
        """
        if not self.is_loaded:
            return {
                "status": "not_loaded",
                "message": "Model not loaded",
            }

        return {
            "status": "loaded",
            "version": self.model_metadata.get("version", "unknown"),
            "trained_date": self.model_metadata.get("trained_date", "unknown"),
            "features": self.feature_definitions.get_feature_names(),
            "model_type": type(self.model).__name__,
            "performance_metrics": self.model_metadata.get("metrics", {}),
        }

    def reload_model(self) -> bool:
        """
        Reload the model from disk.

        Useful for loading updated models without restarting the server.

        Returns:
            True if successful, False otherwise
        """
        logger.info("Reloading model...")
        self.is_loaded = False
        return self.load_model()


# Singleton instance
_predictor: Optional[NeurodevelopmentalPredictor] = None


def get_predictor() -> NeurodevelopmentalPredictor:
    """
    Get or create the predictor instance (singleton pattern).

    Returns:
        NeurodevelopmentalPredictor instance
    """
    global _predictor
    if _predictor is None:
        _predictor = NeurodevelopmentalPredictor()
        _predictor.load_model()
    return _predictor


def reload_predictor() -> bool:
    """
    Reload the predictor with updated model.

    Returns:
        True if successful, False otherwise
    """
    predictor = get_predictor()
    return predictor.reload_model()


def load_model() -> NeurodevelopmentalPredictor:
    """
    Load model function for backward compatibility.

    Returns:
        NeurodevelopmentalPredictor instance
    """
    return get_predictor()


# Backward compatibility alias
NeuroriskPredictor = NeurodevelopmentalPredictor
