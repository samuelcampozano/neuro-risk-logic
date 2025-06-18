"""
Pydantic schemas for API response models.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class PredictionResponse(BaseModel):
    """Response model for risk predictions."""
    
    risk_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Risk probability (0.0 to 1.0)"
    )
    risk_level: str = Field(
        ..., 
        description="Risk category (low/moderate/high)"
    )
    confidence_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Model confidence in prediction"
    )
    risk_factors: List[str] = Field(
        ..., 
        description="Key risk factors identified"
    )
    protective_factors: List[str] = Field(
        ..., 
        description="Protective factors identified"
    )
    feature_importance: Optional[Dict[str, float]] = Field(
        None,
        description="Feature contributions to prediction"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Clinical recommendations based on risk"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "risk_score": 0.72,
                "risk_level": "high",
                "confidence_score": 0.89,
                "risk_factors": [
                    "Family history of neurological disorders",
                    "Psychiatric diagnosis present",
                    "Limited social support"
                ],
                "protective_factors": [
                    "Access to healthcare",
                    "No substance use"
                ],
                "feature_importance": {
                    "family_neuro_history": 0.25,
                    "psychiatric_diagnosis": 0.18,
                    "social_support_level": 0.12
                },
                "recommendations": [
                    "Comprehensive neurological evaluation recommended",
                    "Consider psychiatric support services",
                    "Enhance social support network"
                ]
            }
        }
    )


class AssessmentResponse(BaseModel):
    """Response model for completed assessments."""
    
    success: bool = Field(True, description="Operation success status")
    assessment_id: int = Field(..., description="Database ID of saved assessment")
    prediction: PredictionResponse = Field(..., description="Risk prediction results")
    model_version: str = Field(..., description="Version of model used")
    timestamp: datetime = Field(..., description="Assessment timestamp")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "assessment_id": 123,
                "prediction": {
                    "risk_score": 0.35,
                    "risk_level": "moderate",
                    "confidence_score": 0.92,
                    "risk_factors": ["Birth complications"],
                    "protective_factors": ["Good social support", "Healthcare access"],
                    "recommendations": ["Regular monitoring recommended"]
                },
                "model_version": "v1.0.0",
                "timestamp": "2024-01-15T10:30:00"
            }
        }
    )


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Detailed error message")
    status_code: int = Field(..., description="HTTP status code")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "ValidationError",
                "detail": "Age must be between 0 and 120",
                "status_code": 400,
                "timestamp": "2024-01-15T10:30:00"
            }
        }
    )


class TokenResponse(BaseModel):
    """Response model for authentication."""
    
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    scopes: List[str] = Field(..., description="Granted permission scopes")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 1800,
                "scopes": ["read", "write", "admin"]
            }
        }
    )


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    
    ml_model_type: str = Field(..., description="Type of ML model", alias="model_type")
    ml_model_version: str = Field(..., description="Current model version", alias="model_version")
    features_count: int = Field(..., description="Number of input features")
    training_date: datetime = Field(..., description="When model was trained")
    performance_metrics: Dict[str, float] = Field(..., description="Model performance metrics")
    feature_definitions: List[Dict[str, Any]] = Field(..., description="Feature specifications")
    is_loaded: bool = Field(..., description="Whether model is loaded in memory")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_type": "RandomForestClassifier",
                "model_version": "v1.0.0",
                "features_count": 18,
                "training_date": "2024-01-10T15:30:00",
                "performance_metrics": {
                    "accuracy": 0.85,
                    "precision": 0.82,
                    "recall": 0.88,
                    "f1_score": 0.85,
                    "auc_roc": 0.91
                },
                "feature_definitions": [
                    {"name": "consanguinity", "type": "binary", "importance": 0.15},
                    {"name": "family_neuro_history", "type": "binary", "importance": 0.12}
                ],
                "is_loaded": True
            }
        }
    )


class StatsResponse(BaseModel):
    """Response model for system statistics."""
    
    total_assessments: int = Field(..., description="Total number of assessments")
    assessments_by_risk_level: Dict[str, int] = Field(..., description="Distribution by risk level")
    assessments_by_gender: Dict[str, int] = Field(..., description="Distribution by gender")
    average_risk_score: float = Field(..., description="Average risk score")
    average_age: float = Field(..., description="Average subject age")
    most_common_risk_factors: List[Dict[str, Any]] = Field(..., description="Top risk factors")
    assessments_last_30_days: int = Field(..., description="Recent assessment count")
    ml_performance: Dict[str, float] = Field(..., description="Current model metrics", alias="model_performance")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_assessments": 1523,
                "assessments_by_risk_level": {
                    "low": 612,
                    "moderate": 687,
                    "high": 224
                },
                "assessments_by_gender": {
                    "M": 798,
                    "F": 705,
                    "Other": 20
                },
                "average_risk_score": 0.42,
                "average_age": 31.5,
                "most_common_risk_factors": [
                    {"factor": "family_neuro_history", "frequency": 0.35},
                    {"factor": "psychiatric_diagnosis", "frequency": 0.28},
                    {"factor": "extreme_poverty", "frequency": 0.22}
                ],
                "assessments_last_30_days": 145,
                "model_performance": {
                    "current_accuracy": 0.86,
                    "current_auc": 0.92
                }
            }
        }
    )


class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint."""
    
    status: str = Field(..., description="Service health status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    services: Dict[str, Dict[str, Any]] = Field(..., description="Individual service statuses")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2024-01-15T10:30:00",
                "services": {
                    "database": {
                        "status": "connected",
                        "response_time_ms": 5
                    },
                    "ml_model": {
                        "status": "loaded",
                        "version": "v1.0.0",
                        "memory_usage_mb": 150
                    },
                    "cache": {
                        "status": "active",
                        "hit_rate": 0.85
                    }
                }
            }
        }
    )