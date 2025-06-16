"""
SQLAlchemy model for storing neurodevelopmental risk assessments.
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, 
    DateTime, JSON, Text, Index
)
from sqlalchemy.ext.hybrid import hybrid_property
from app.database import Base


class Assessment(Base):
    """
    Model to store neurodevelopmental risk assessments.
    
    This model captures:
    - Clinical and genetic risk factors
    - Sociodemographic indicators
    - ML model predictions
    - Metadata for tracking and analysis
    """
    
    __tablename__ = "assessments"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # Demographics
    age = Column(Integer, nullable=False, comment="Subject's age in years")
    gender = Column(String(10), nullable=False, comment="Subject's gender (M/F/Other)")
    
    # Clinical-Genetic Features (Binary)
    consanguinity = Column(Boolean, nullable=False, comment="Parents are blood-related")
    family_neuro_history = Column(Boolean, nullable=False, comment="Family history of neurological disorders")
    seizures_history = Column(Boolean, nullable=False, comment="History of seizures or convulsions")
    brain_injury_history = Column(Boolean, nullable=False, comment="History of traumatic brain injury")
    psychiatric_diagnosis = Column(Boolean, nullable=False, comment="Diagnosed with psychiatric disorder")
    substance_use = Column(Boolean, nullable=False, comment="Regular psychoactive substance use")
    suicide_ideation = Column(Boolean, nullable=False, comment="History of suicide attempts or ideation")
    psychotropic_medication = Column(Boolean, nullable=False, comment="Currently on psychotropic medication")
    
    # Sociodemographic Features (Binary)
    birth_complications = Column(Boolean, nullable=False, comment="Born with complications or low weight")
    extreme_poverty = Column(Boolean, nullable=False, comment="Lives in extreme poverty")
    education_access_issues = Column(Boolean, nullable=False, comment="Restricted access to education")
    healthcare_access = Column(Boolean, nullable=False, comment="Has access to mental healthcare")
    disability_diagnosis = Column(Boolean, nullable=False, comment="Diagnosed with a disability")
    breastfed_infancy = Column(Boolean, nullable=False, comment="Was breastfed during infancy")
    violence_exposure = Column(Boolean, nullable=False, comment="Exposed to violence or trauma")
    
    # Categorical Feature
    social_support_level = Column(
        String(20), 
        nullable=False, 
        comment="Level of social support (isolated/moderate/supported)"
    )
    
    # Risk Assessment Results
    risk_score = Column(
        Float, 
        nullable=False, 
        comment="Calculated risk score (0.0 to 1.0)"
    )
    risk_level = Column(
        String(20), 
        nullable=False, 
        comment="Risk category (low/moderate/high)"
    )
    confidence_score = Column(
        Float, 
        nullable=True, 
        comment="Model confidence in prediction (0.0 to 1.0)"
    )
    
    # Feature importance snapshot
    feature_contributions = Column(
        JSON, 
        nullable=True, 
        comment="Individual feature contributions to risk score"
    )
    
    # Metadata
    model_version = Column(
        String(50), 
        nullable=False, 
        comment="Version of ML model used"
    )
    assessment_date = Column(
        DateTime, 
        default=datetime.utcnow, 
        nullable=False, 
        comment="Timestamp of assessment"
    )
    
    # Optional fields for tracking
    clinician_id = Column(
        String(100), 
        nullable=True, 
        comment="ID of clinician who requested assessment"
    )
    notes = Column(
        Text, 
        nullable=True, 
        comment="Additional clinical notes"
    )
    
    # Consent and compliance
    consent_given = Column(
        Boolean, 
        default=False, 
        nullable=False, 
        comment="Subject consent for data use"
    )
    
    # Create indexes for common queries
    __table_args__ = (
        Index('idx_risk_level', 'risk_level'),
        Index('idx_assessment_date', 'assessment_date'),
        Index('idx_age_gender', 'age', 'gender'),
    )
    
    @hybrid_property
    def feature_vector(self) -> list:
        """
        Get feature vector for ML model input.
        Returns features in the order expected by the model.
        """
        return [
            # Binary features (converted to int)
            int(self.consanguinity),
            int(self.family_neuro_history),
            int(self.seizures_history),
            int(self.brain_injury_history),
            int(self.psychiatric_diagnosis),
            int(self.substance_use),
            int(self.birth_complications),
            int(self.extreme_poverty),
            int(self.education_access_issues),
            int(self.healthcare_access),
            int(self.suicide_ideation),
            int(self.psychotropic_medication),
            int(self.disability_diagnosis),
            int(self.breastfed_infancy),
            int(self.violence_exposure),
            # Categorical encoding for social support
            self._encode_social_support(),
            # Numeric features
            self.age,
            self._encode_gender()
        ]
    
    def _encode_social_support(self) -> int:
        """Encode social support level as numeric."""
        mapping = {
            "isolated": 0,
            "moderate": 1,
            "supported": 2
        }
        return mapping.get(self.social_support_level.lower(), 1)
    
    def _encode_gender(self) -> int:
        """Encode gender as numeric."""
        mapping = {
            "M": 0,
            "F": 1,
            "Other": 2
        }
        return mapping.get(self.gender.upper(), 2)
    
    def to_dict(self) -> dict:
        """Convert model instance to dictionary."""
        return {
            "id": self.id,
            "demographics": {
                "age": self.age,
                "gender": self.gender
            },
            "clinical_factors": {
                "consanguinity": self.consanguinity,
                "family_neuro_history": self.family_neuro_history,
                "seizures_history": self.seizures_history,
                "brain_injury_history": self.brain_injury_history,
                "psychiatric_diagnosis": self.psychiatric_diagnosis,
                "substance_use": self.substance_use,
                "suicide_ideation": self.suicide_ideation,
                "psychotropic_medication": self.psychotropic_medication
            },
            "sociodemographic_factors": {
                "birth_complications": self.birth_complications,
                "extreme_poverty": self.extreme_poverty,
                "education_access_issues": self.education_access_issues,
                "healthcare_access": self.healthcare_access,
                "disability_diagnosis": self.disability_diagnosis,
                "social_support_level": self.social_support_level,
                "breastfed_infancy": self.breastfed_infancy,
                "violence_exposure": self.violence_exposure
            },
            "risk_assessment": {
                "risk_score": self.risk_score,
                "risk_level": self.risk_level,
                "confidence_score": self.confidence_score,
                "model_version": self.model_version
            },
            "metadata": {
                "assessment_date": self.assessment_date.isoformat() if self.assessment_date else None,
                "clinician_id": self.clinician_id,
                "consent_given": self.consent_given
            }
        }
    
    def __repr__(self):
        return (
            f"<Assessment(id={self.id}, "
            f"age={self.age}, "
            f"gender='{self.gender}', "
            f"risk_level='{self.risk_level}', "
            f"risk_score={self.risk_score:.3f})>"
        )