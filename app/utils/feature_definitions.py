"""
Utility functions for managing feature definitions and transformations.
"""

import json
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import numpy as np

from app.config import settings
from loguru import logger


class FeatureDefinitions:
    """Manages feature definitions and transformations."""
    
    def __init__(self, definitions_path: Optional[Path] = None):
        """
        Initialize feature definitions manager.
        
        Args:
            definitions_path: Path to feature definitions JSON file
        """
        self.definitions_path = definitions_path or settings.get_feature_definitions_path()
        self.features = {}
        self.feature_order = []
        self.risk_thresholds = {}
        self.model_config = {}
        self._load_definitions()
    
    def _load_definitions(self):
        """Load feature definitions from JSON file."""
        try:
            with open(self.definitions_path, 'r') as f:
                data = json.load(f)
            
            # Store features by name for easy access
            for feature in data['features']:
                self.features[feature['name']] = feature
                self.feature_order.append(feature['name'])
            
            self.risk_thresholds = data.get('risk_thresholds', {})
            self.model_config = data.get('model_config', {})
            
            logger.info(f"Loaded {len(self.features)} feature definitions")
            
        except Exception as e:
            logger.error(f"Error loading feature definitions: {str(e)}")
            raise
    
    def get_feature_vector(self, assessment_data: Dict[str, Any]) -> List[float]:
        """
        Convert assessment data to feature vector for model input.
        
        Args:
            assessment_data: Dictionary with assessment field values
            
        Returns:
            List of feature values in the correct order
        """
        feature_vector = []
        
        for feature_name in self.feature_order:
            feature_def = self.features[feature_name]
            value = assessment_data.get(feature_name)
            
            if value is None:
                raise ValueError(f"Missing required feature: {feature_name}")
            
            # Transform based on feature type
            if feature_def['type'] == 'binary':
                feature_vector.append(float(value))
            
            elif feature_def['type'] == 'categorical':
                # Use encoding mapping
                encoding = feature_def.get('encoding', {})
                if isinstance(value, str):
                    encoded_value = encoding.get(value.lower(), 0)
                else:
                    encoded_value = value
                feature_vector.append(float(encoded_value))
            
            elif feature_def['type'] == 'numeric':
                # Normalize numeric features if range is specified
                if 'range' in feature_def:
                    min_val, max_val = feature_def['range']
                    normalized = (value - min_val) / (max_val - min_val)
                    feature_vector.append(normalized)
                else:
                    feature_vector.append(float(value))
            
            else:
                feature_vector.append(float(value))
        
        return feature_vector
    
    def get_feature_names(self) -> List[str]:
        """Get ordered list of feature names."""
        return self.feature_order.copy()
    
    def get_feature_info(self, feature_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific feature."""
        return self.features.get(feature_name)
    
    def get_risk_level(self, risk_score: float) -> str:
        """
        Convert risk score to risk level category.
        
        Args:
            risk_score: Numerical risk score (0-1)
            
        Returns:
            Risk level string (low/moderate/high)
        """
        for level, (min_score, max_score) in self.risk_thresholds.items():
            if min_score <= risk_score < max_score:
                return level
        return "unknown"
    
    def get_high_risk_features(self) -> List[str]:
        """Get list of features with positive risk direction."""
        return [
            name for name, info in self.features.items()
            if info.get('risk_direction') == 'positive'
        ]
    
    def get_protective_features(self) -> List[str]:
        """Get list of features with negative risk direction."""
        return [
            name for name, info in self.features.items()
            if info.get('risk_direction') == 'negative'
        ]
    
    def calculate_feature_contributions(
        self, 
        feature_values: Dict[str, Any],
        feature_importances: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Calculate individual feature contributions to risk.
        
        Args:
            feature_values: Dictionary of feature values
            feature_importances: Optional ML model feature importances
            
        Returns:
            Dictionary of feature contributions
        """
        contributions = {}
        
        for feature_name, value in feature_values.items():
            if feature_name not in self.features:
                continue
            
            feature_def = self.features[feature_name]
            base_weight = feature_def.get('weight', 0.1)
            
            # Use model importance if available, otherwise use predefined weight
            if feature_importances and feature_name in feature_importances:
                weight = feature_importances[feature_name]
            else:
                weight = base_weight
            
            # Calculate contribution based on value and risk direction
            if feature_def['type'] == 'binary':
                contribution = weight if value else 0
            elif feature_def['type'] == 'categorical':
                # Normalize categorical contribution
                encoding = feature_def.get('encoding', {})
                max_encoding = max(encoding.values()) if encoding else 1
                norm_value = value / max_encoding if max_encoding > 0 else 0
                contribution = weight * norm_value
            else:
                # For numeric features
                contribution = weight * (value / 100.0)  # Assuming percentage
            
            # Adjust for risk direction
            if feature_def.get('risk_direction') == 'negative':
                contribution = -contribution
            
            contributions[feature_name] = contribution
        
        return contributions


# Singleton instance
_feature_definitions: Optional[FeatureDefinitions] = None


def load_feature_definitions(force_reload: bool = False) -> FeatureDefinitions:
    """
    Load feature definitions (singleton pattern).
    
    Args:
        force_reload: Force reload from file
        
    Returns:
        FeatureDefinitions instance
    """
    global _feature_definitions
    
    if _feature_definitions is None or force_reload:
        _feature_definitions = FeatureDefinitions()
    
    return _feature_definitions


def get_feature_vector(assessment_data: Dict[str, Any]) -> List[float]:
    """
    Convenience function to get feature vector.
    
    Args:
        assessment_data: Assessment data dictionary
        
    Returns:
        Feature vector for model input
    """
    definitions = load_feature_definitions()
    return definitions.get_feature_vector(assessment_data)


def get_feature_importance(
    feature_values: Dict[str, Any],
    model_importances: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Convenience function to get feature importance/contributions.
    
    Args:
        feature_values: Dictionary of feature values
        model_importances: Optional model-derived importances
        
    Returns:
        Feature contribution dictionary
    """
    definitions = load_feature_definitions()
    return definitions.calculate_feature_contributions(
        feature_values, 
        model_importances
    )