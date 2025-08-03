"""
Pydantic schemas for API request validation.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, validator, ConfigDict


class AssessmentRequest(BaseModel):
    """
    Schema for submitting a new neurodevelopmental risk assessment.
    Includes all 18 features identified for the model.
    """

    # Demographics
    age: int = Field(..., ge=0, le=120, description="Subject's age in years")
    gender: Literal["M", "F", "Other"] = Field(..., description="Subject's gender")

    # Clinical-Genetic Features (Binary)
    consanguinity: bool = Field(..., description="Are the subject's parents blood-related?")
    family_neuro_history: bool = Field(..., description="Family history of neurological disorders?")
    seizures_history: bool = Field(
        ..., description="Has the subject ever had seizures or convulsions?"
    )
    brain_injury_history: bool = Field(
        ..., description="History of traumatic brain injury (TBI) or head trauma?"
    )
    psychiatric_diagnosis: bool = Field(..., description="Diagnosed with any psychiatric disorder?")
    substance_use: bool = Field(
        ..., description="Does the subject consume psychoactive substances regularly?"
    )
    suicide_ideation: bool = Field(..., description="History of suicide attempts or ideation?")
    psychotropic_medication: bool = Field(
        ..., description="Is the subject on psychotropic or neurological medication?"
    )

    # Sociodemographic Features (Binary)
    birth_complications: bool = Field(
        ..., description="Was the subject born with complications or low weight?"
    )
    extreme_poverty: bool = Field(..., description="Does the subject live in extreme poverty?")
    education_access_issues: bool = Field(
        ..., description="Has the subject ever had access restrictions to education?"
    )
    healthcare_access: bool = Field(
        ..., description="Does the subject have access to mental healthcare?"
    )
    disability_diagnosis: bool = Field(
        ..., description="Has the subject ever been diagnosed with a disability?"
    )
    breastfed_infancy: bool = Field(..., description="Was the subject breastfed during infancy?")
    violence_exposure: bool = Field(
        ..., description="Exposure to violence or trauma (childhood or adulthood)?"
    )

    # Categorical Feature
    social_support_level: Literal["isolated", "moderate", "supported"] = Field(
        ..., description="Level of social support"
    )

    # Optional metadata
    clinician_id: Optional[str] = Field(None, description="ID of clinician requesting assessment")
    notes: Optional[str] = Field(None, max_length=1000, description="Additional clinical notes")
    consent_given: bool = Field(..., description="Subject consent for data use")

    @validator("gender")
    def validate_gender(cls, v):
        """Ensure gender is uppercase."""
        return v.upper()

    @validator("social_support_level")
    def validate_social_support(cls, v):
        """Ensure social support level is lowercase."""
        return v.lower()

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
                "consent_given": True,
                "clinician_id": "DR12345",
                "notes": "Patient referred for comprehensive assessment",
            }
        }
    )


class PredictionRequest(BaseModel):
    """
    Schema for making a prediction without storing data.
    Similar to AssessmentRequest but without metadata fields.
    """

    # Demographics
    age: int = Field(..., ge=0, le=120)
    gender: Literal["M", "F", "Other"]

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
    social_support_level: Literal["isolated", "moderate", "supported"]
    breastfed_infancy: bool
    violence_exposure: bool

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 30,
                "gender": "F",
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
                "social_support_level": "supported",
                "breastfed_infancy": True,
                "violence_exposure": False,
            }
        }
    )


class LoginRequest(BaseModel):
    """Schema for authentication login."""

    api_key: str = Field(..., min_length=10, description="API key for authentication")

    model_config = ConfigDict(json_schema_extra={"example": {"api_key": "your-api-key-here"}})
