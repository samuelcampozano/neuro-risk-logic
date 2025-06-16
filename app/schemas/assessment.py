"""
Pydantic schemas for Assessment model CRUD operations.
"""

from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class AssessmentBase(BaseModel):
    """Base schema with common assessment fields."""
    
    # Demographics
    age: int = Field(..., ge=0, le=120)
    gender: str = Field(..., max_length=10)
    
    # Clinical-Genetic Features
    consanguinity: bool
    family_neuro_history: bool
    seizures_history: bool
    brain_injury_history: bool
    psychiatric_diagnosis: bool
    substance_use: bool
    suicide_ideation: bool
    psychotropic_medication: bool
    
    # Sociodemographic Features
    birth_complications: bool
    extreme_poverty: bool
    education_access_issues: bool
    healthcare_access: bool
    disability_diagnosis: bool
    social_support_level: str
    breastfed_infancy: bool
    violence_exposure: bool
    
    # Optional fields
    clinician_id: Optional[str] = None
    notes: Optional[str] = None
    consent_given: bool = True


class AssessmentCreate(AssessmentBase):
    """Schema for creating a new assessment."""
    pass


class AssessmentUpdate(BaseModel):
    """Schema for updating an assessment (all fields optional)."""
    
    # Allow partial updates
    age: Optional[int] = Field(None, ge=0, le=120)
    gender: Optional[str] = Field(None, max_length=10)
    
    # Clinical-Genetic Features
    consanguinity: Optional[bool] = None
    family_neuro_history: Optional[bool] = None
    seizures_history: Optional[bool] = None
    brain_injury_history: Optional[bool] = None
    psychiatric_diagnosis: Optional[bool] = None
    substance_use: Optional[bool] = None
    suicide_ideation: Optional[bool] = None
    psychotropic_medication: Optional[bool] = None
    
    # Sociodemographic Features
    birth_complications: Optional[bool] = None
    extreme_poverty: Optional[bool] = None
    education_access_issues: Optional[bool] = None
    healthcare_access: Optional[bool] = None
    disability_diagnosis: Optional[bool] = None
    social_support_level: Optional[str] = None
    breastfed_infancy: Optional[bool] = None
    violence_exposure: Optional[bool] = None
    
    # Metadata
    clinician_id: Optional[str] = None
    notes: Optional[str] = None


class AssessmentInDB(AssessmentBase):
    """Schema for assessment stored in database."""
    
    id: int
    risk_score: float
    risk_level: str
    confidence_score: Optional[float]
    feature_contributions: Optional[Dict[str, float]]
    model_version: str
    assessment_date: datetime
    
    model_config = ConfigDict(from_attributes=True)


class AssessmentDetail(AssessmentInDB):
    """Detailed assessment view with additional computed fields."""
    
    risk_factors: List[str] = Field(default_factory=list)
    protective_factors: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 123,
                "age": 25,
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
                "violence_exposure": False,
                "risk_score": 0.65,
                "risk_level": "moderate",
                "confidence_score": 0.87,
                "feature_contributions": {
                    "family_neuro_history": 0.15,
                    "psychiatric_diagnosis": 0.12
                },
                "model_version": "v1.0.0",
                "assessment_date": "2024-01-15T10:30:00",
                "clinician_id": "DR12345",
                "consent_given": True,
                "risk_factors": [
                    "Family history of neurological disorders",
                    "Existing psychiatric diagnosis"
                ],
                "protective_factors": [
                    "Access to healthcare",
                    "Moderate social support"
                ],
                "recommendations": [
                    "Regular psychiatric follow-up",
                    "Neurological screening recommended"
                ]
            }
        }
    )