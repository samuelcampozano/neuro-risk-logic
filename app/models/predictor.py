"""
Machine Learning predictor for neurodevelopmental risk assessment.
"""

import os
import pickle
import joblib
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator

from app.config import settings
from app.utils.feature_definitions import load_feature_definitions
from app.utils.risk_calculator import (
    calculate_risk_factors,
    generate_recommendations,
    interpret_risk_level
)
from loguru import logger


class NeuroriskPredictor:
    """ML predictor for neurodevelopmental risk assessment."""
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model file
        """
        self.model_path = model_path or settings.get_model_path()
        self.model: Optional[BaseEstimator] = None
        self.model_metadata: Dict[str, Any] = {}
        self.feature_definitions = load_feature_definitions()
        self.is_loaded = False
        
    def load_model(self) -> bool:
        """
        Load trained model from disk.
        
        Returns:
            bool: True if successful
        """
        try:
            if not self.model_path.exists():
                logger.warning(f"Model file not found at {self.model_path}")
                return False
            
            # Try joblib first (preferred for sklearn models)
            try:
                self.model = joblib.load(self.model_path)
            except:
                # Fallback to pickle
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
            
            # Load metadata if available
            metadata_path = self.model_path.with_suffix('.json')
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
            
            self.is_loaded = True
            logger.info(f"Model loaded successfully from {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def predict(
        self,
        assessment_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Make risk prediction for assessment data.
        
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
            confidence = max(
                self.model.predict_proba(feature_array)[0]
            )
            
            # Get feature importances if available
            feature_importances = {}
            if hasattr(self.model, 'feature_importances_'):
                feature_names = self.feature_definitions.get_feature_names()
                for idx, importance in enumerate(self.model.feature_importances_):
                    if idx < len(feature_names):
                        feature_importances[feature_names[idx]] = float(importance)
            
            # Calculate risk factors and protective factors
            risk_factors, protective_factors = calculate_risk_factors(
                assessment_data,
                feature_importances
            )
            
            # Get risk level
            risk_level = self.feature_definitions.get_risk_level(risk_probability)
            
            # Generate recommendations
            recommendations = generate_recommendations(
                risk_level,
                risk_factors,
                assessment_data
            )
            
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
                "prediction_timestamp": datetime.utcnow().isoformat()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            "is_loaded": self.is_loaded,
            "model_path": str(self.model_path),
            "model_version": settings.model_version,
            "features_count": len(self.feature_definitions.features)
        }
        
        if self.is_loaded and self.model:
            info.update({
                "model_type": type(self.model).__name__,
                "model_params": self.model.get_params() if hasattr(self.model, 'get_params') else {},
                "feature_names": self.feature_definitions.get_feature_names()
            })
            
            if hasattr(self.model, 'n_estimators'):
                info["n_estimators"] = self.model.n_estimators
            
            if hasattr(self.model, 'feature_importances_'):
                info["has_feature_importances"] = True
        
        # Add metadata
        info.update(self.model_metadata)
        
        return info
    
    def validate_model(self) -> Tuple[bool, Optional[str]]:
        """
        Validate that model is properly configured.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if not self.is_loaded:
                if not self.load_model():
                    return False, "Failed to load model"
            
            # Test with dummy data
            test_data = self._generate_test_data()
            result = self.predict(test_data)
            
            # Validate result structure
            required_keys = {
                "risk_score", "risk_level", "confidence_score",
                "risk_factors", "protective_factors", "recommendations"
            }
            
            missing_keys = required_keys - set(result.keys())
            if missing_keys:
                return False, f"Missing result keys: {missing_keys}"
            
            # Validate value ranges
            if not 0 <= result["risk_score"] <= 1:
                return False, "Risk score out of range"
            
            if not 0 <= result["confidence_score"] <= 1:
                return False, "Confidence score out of range"
            
            if result["risk_level"] not in ["low", "moderate", "high"]:
                return False, f"Invalid risk level: {result['risk_level']}"
            
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def _generate_test_data(self) -> Dict[str, Any]:
        """Generate test data for validation."""
        return {
            "age": 30,
            "gender": "M",
            "consanguinity": False,
            "family_neuro_history": True,
            "seizures_history": False,
            "brain_injury_history": False,
            "psychiatric_diagnosis": True,
            "substance_use": False,
            "suicide_ideation": False,
            "psychotropic_medication": True,
            "birth_complications": False,
            "extreme_poverty": False,
            "education_access_issues": False,
            "healthcare_access": True,
            "disability_diagnosis": False,
            "social_support_level": "moderate",
            "breastfed_infancy": True,
            "violence_exposure": False
        }


# Global predictor instance
_predictor: Optional[NeuroriskPredictor] = None


def load_model(model_path: Optional[Path] = None) -> NeuroriskPredictor:
    """
    Load or get the global predictor instance.
    
    Args:
        model_path: Optional custom model path
        
    Returns:
        NeuroriskPredictor instance
    """
    global _predictor
    
    if _predictor is None or model_path:
        _predictor = NeuroriskPredictor(model_path)
        _predictor.load_model()
    
    return _predictor


def get_predictor() -> NeuroriskPredictor:
    """
    Get the global predictor instance.
    
    Returns:
        NeuroriskPredictor instance
    """
    return load_model()


def reload_predictor() -> bool:
    """
    Reload the model (useful after retraining).
    
    Returns:
        bool: True if successful
    """
    global _predictor
    _predictor = None
    predictor = load_model()
    return predictor.is_loaded