"""
Pydantic schemas for request/response validation.
"""

from app.schemas.request import AssessmentRequest, PredictionRequest, LoginRequest
from app.schemas.response import (
    AssessmentResponse,
    PredictionResponse,
    ErrorResponse,
    TokenResponse,
    ModelInfoResponse,
    StatsResponse,
)
from app.schemas.assessment import (
    AssessmentBase,
    AssessmentCreate,
    AssessmentUpdate,
    AssessmentInDB,
    AssessmentDetail,
)

__all__ = [
    # Request schemas
    "AssessmentRequest",
    "PredictionRequest",
    "LoginRequest",
    # Response schemas
    "AssessmentResponse",
    "PredictionResponse",
    "ErrorResponse",
    "TokenResponse",
    "ModelInfoResponse",
    "StatsResponse",
    # Assessment schemas
    "AssessmentBase",
    "AssessmentCreate",
    "AssessmentUpdate",
    "AssessmentInDB",
    "AssessmentDetail",
]
