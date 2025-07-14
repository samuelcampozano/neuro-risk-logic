"""
Risk calculation and interpretation utilities.
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np

from app.utils.feature_definitions import load_feature_definitions
from loguru import logger


class RiskCalculator:
    """Handles risk calculation and interpretation logic."""

    def __init__(self):
        """Initialize risk calculator with feature definitions."""
        self.definitions = load_feature_definitions()

    def calculate_risk_factors(
        self,
        assessment_data: Dict[str, Any],
        feature_contributions: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Identify risk factors and protective factors from assessment.

        Args:
            assessment_data: Assessment data dictionary
            feature_contributions: Optional feature contribution scores

        Returns:
            Tuple of (risk_factors, protective_factors)
        """
        risk_factors = []
        protective_factors = []

        # Get high-risk and protective feature lists
        high_risk_features = self.definitions.get_high_risk_features()
        protective_features = self.definitions.get_protective_features()

        # Check each feature
        for feature_name, value in assessment_data.items():
            if feature_name not in self.definitions.features:
                continue

            feature_info = self.definitions.get_feature_info(feature_name)

            # For binary features
            if feature_info["type"] == "binary":
                if value and feature_name in high_risk_features:
                    risk_factors.append(feature_info["display_name"])
                elif value and feature_name in protective_features:
                    protective_factors.append(feature_info["display_name"])
                elif not value and feature_name in protective_features:
                    # Not having a protective factor is a risk
                    risk_factors.append(f"Lack of {feature_info['display_name']}")

            # For categorical features
            elif feature_info["type"] == "categorical":
                if feature_name == "social_support_level":
                    if value == "isolated":
                        risk_factors.append("Social isolation")
                    elif value == "supported":
                        protective_factors.append("Strong social support")

                elif feature_name == "gender":
                    # Gender-specific risk patterns could be added here
                    pass

            # For numeric features
            elif feature_info["type"] == "numeric":
                if feature_name == "age":
                    # Age-specific risk patterns
                    if value < 5:
                        risk_factors.append(
                            "Very young age (early developmental stage)"
                        )
                    elif value > 65:
                        risk_factors.append("Advanced age")

        # Sort by contribution if available
        if feature_contributions:
            risk_factors.sort(
                key=lambda x: abs(feature_contributions.get(x, 0)), reverse=True
            )
            protective_factors.sort(
                key=lambda x: abs(feature_contributions.get(x, 0)), reverse=True
            )

        # Always return at least a message for protective factors
        if not protective_factors:
            protective_factors = ["No protective factors were identified"]

        return risk_factors[:5], protective_factors[:5]  # Top 5 each

    def generate_recommendations(
        self,
        risk_level: str,
        risk_factors: List[str],
        assessment_data: Dict[str, Any],
    ) -> List[str]:
        """
        Generate clinical recommendations based on risk assessment.

        Args:
            risk_level: Risk level (low/moderate/high)
            risk_factors: List of identified risk factors
            assessment_data: Full assessment data

        Returns:
            List of recommendations
        """
        recommendations = []

        # Base recommendations by risk level
        if risk_level == "high":
            recommendations.append(
                "Comprehensive mental health evaluation recommended"
            )
            recommendations.append("Priority referral to specialist care")
        elif risk_level == "moderate":
            recommendations.append(
                "Mental health screening recommended within 3 months"
            )
            recommendations.append("Regular monitoring of symptoms")
        else:
            recommendations.append("Continue routine health monitoring")

        # Specific recommendations based on risk factors
        if assessment_data.get("family_neuro_history"):
            recommendations.append("Genetic counseling may be beneficial")

        if assessment_data.get("psychiatric_diagnosis"):
            recommendations.append(
                "Ensure ongoing psychiatric care and medication compliance"
            )

        if assessment_data.get("seizures_history"):
            recommendations.append("Neurological evaluation with EEG recommended")

        if assessment_data.get("suicide_ideation"):
            recommendations.append("Psychiatric evaluation recommended")
            recommendations.append("Implement support and monitoring protocol")

        if assessment_data.get("substance_use"):
            recommendations.append("Substance abuse counseling and support services")

        if assessment_data.get("extreme_poverty"):
            recommendations.append("Connect with social services for support programs")

        if not assessment_data.get("healthcare_access"):
            recommendations.append("Facilitate access to mental health services")

        if assessment_data.get("social_support_level") == "isolated":
            recommendations.append(
                "Develop social support network and community connections"
            )

        if assessment_data.get("violence_exposure"):
            recommendations.append("Trauma-informed therapy recommended")

        # Educational recommendations
        if assessment_data.get("education_access_issues"):
            recommendations.append(
                "Educational support and cognitive stimulation programs"
            )

        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)

        return unique_recommendations

    def interpret_risk_level(
        self, risk_score: float, confidence_score: float
    ) -> Dict[str, Any]:
        """
        Provide detailed interpretation of risk assessment.

        Args:
            risk_score: Numerical risk score (0-1)
            confidence_score: Model confidence (0-1)

        Returns:
            Dictionary with interpretation details
        """
        risk_level = self.definitions.get_risk_level(risk_score)

        interpretations = {
            "low": {
                "summary": "Low risk of mental health alterations",
                "description": "Current assessment indicates minimal risk factors. Continue standard care and monitoring.",
                "action_priority": "routine",
                "follow_up_months": 12,
            },
            "moderate": {
                "summary": "Moderate risk of mental health alterations",
                "description": "Several risk factors identified that warrant closer monitoring and possible intervention.",
                "action_priority": "elevated",
                "follow_up_months": 3,
            },
            "high": {
                "summary": "High risk of mental health alterations",
                "description": "Multiple significant risk factors present. Evaluation and intervention recommended.",
                "action_priority": "priority",
                "follow_up_months": 1,
            },
        }

        interpretation = interpretations.get(risk_level, interpretations["moderate"])

        # Adjust based on confidence
        if confidence_score < 0.7:
            interpretation[
                "confidence_note"
            ] = "Model confidence is moderate. Consider additional assessment."
        elif confidence_score < 0.5:
            interpretation[
                "confidence_note"
            ] = "Model confidence is low. Results should be interpreted with caution."
        else:
            interpretation[
                "confidence_note"
            ] = "Model shows high confidence in this assessment."

        interpretation["risk_score"] = risk_score
        interpretation["risk_level"] = risk_level
        interpretation["confidence_score"] = confidence_score

        return interpretation


# Convenience functions
def calculate_risk_factors(
    assessment_data: Dict[str, Any],
    feature_contributions: Optional[Dict[str, float]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Convenience function to calculate risk factors.

    Args:
        assessment_data: Assessment data
        feature_contributions: Optional feature contributions

    Returns:
        Tuple of (risk_factors, protective_factors)
    """
    calculator = RiskCalculator()
    return calculator.calculate_risk_factors(assessment_data, feature_contributions)


def generate_recommendations(
    risk_level: str, risk_factors: List[str], assessment_data: Dict[str, Any]
) -> List[str]:
    """
    Convenience function to generate recommendations.

    Args:
        risk_level: Risk level
        risk_factors: Identified risk factors
        assessment_data: Full assessment data

    Returns:
        List of recommendations
    """
    calculator = RiskCalculator()
    return calculator.generate_recommendations(
        risk_level, risk_factors, assessment_data
    )


def interpret_risk_level(
    risk_score: float, confidence_score: float
) -> Dict[str, Any]:
    """
    Convenience function to interpret risk level.

    Args:
        risk_score: Risk score (0-1)
        confidence_score: Confidence score (0-1)

    Returns:
        Interpretation dictionary
    """
    calculator = RiskCalculator()
    return calculator.interpret_risk_level(risk_score, confidence_score)