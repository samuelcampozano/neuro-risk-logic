"""
Routes for submitting and managing assessments.
Mixed authentication - some endpoints public, some protected.
"""

from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_

from app.database import get_db
from app.models.assessment import Assessment
from app.models.predictor import get_predictor
from app.schemas.request import AssessmentRequest
from app.schemas.response import AssessmentResponse, ErrorResponse
from app.schemas.assessment import (
    AssessmentInDB, 
    AssessmentDetail,
    AssessmentUpdate
)
from app.auth import require_token, require_write_permission, TokenData
from loguru import logger

router = APIRouter(
    prefix="/api/v1",
    tags=["assessments"],
    responses={
        401: {"description": "Unauthorized"},
        404: {"description": "Not found"}
    }
)


@router.post(
    "/assessments",
    response_model=AssessmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit new assessment",
    description="Submit a neurodevelopmental risk assessment (PUBLIC endpoint)"
)
async def submit_assessment(
    request: AssessmentRequest,
    db: Session = Depends(get_db)
) -> AssessmentResponse:
    """
    Submit a new assessment for risk calculation and storage.
    
    This is a PUBLIC endpoint - no authentication required.
    Requires explicit consent for data storage.
    
    Args:
        request: Assessment data with consent
        db: Database session
        
    Returns:
        AssessmentResponse with prediction results
    """
    try:
        # Check consent
        if not request.consent_given:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Consent is required to store assessment data"
            )
        
        # Get predictor and make prediction
        predictor = get_predictor()
        assessment_data = request.model_dump()
        prediction_result = predictor.predict(assessment_data)
        
        # Create assessment record
        assessment = Assessment(
            # Demographics
            age=request.age,
            gender=request.gender,
            # Clinical features
            consanguinity=request.consanguinity,
            family_neuro_history=request.family_neuro_history,
            seizures_history=request.seizures_history,
            brain_injury_history=request.brain_injury_history,
            psychiatric_diagnosis=request.psychiatric_diagnosis,
            substance_use=request.substance_use,
            suicide_ideation=request.suicide_ideation,
            psychotropic_medication=request.psychotropic_medication,
            # Sociodemographic features
            birth_complications=request.birth_complications,
            extreme_poverty=request.extreme_poverty,
            education_access_issues=request.education_access_issues,
            healthcare_access=request.healthcare_access,
            disability_diagnosis=request.disability_diagnosis,
            social_support_level=request.social_support_level,
            breastfed_infancy=request.breastfed_infancy,
            violence_exposure=request.violence_exposure,
            # Risk results
            risk_score=prediction_result["risk_score"],
            risk_level=prediction_result["risk_level"],
            confidence_score=prediction_result["confidence_score"],
            feature_contributions=prediction_result.get("feature_importance"),
            # Metadata
            model_version=prediction_result["model_version"],
            clinician_id=request.clinician_id,
            notes=request.notes,
            consent_given=request.consent_given
        )
        
        # Save to database
        db.add(assessment)
        db.commit()
        db.refresh(assessment)
        
        # Prepare response
        response = AssessmentResponse(
            success=True,
            assessment_id=assessment.id,
            prediction=prediction_result,
            model_version=prediction_result["model_version"],
            timestamp=assessment.assessment_date
        )
        
        logger.info(f"Assessment {assessment.id} submitted successfully")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting assessment: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing assessment"
        )


@router.get(
    "/assessments",
    response_model=List[AssessmentInDB],
    summary="List assessments (Protected)",
    description="Get paginated list of assessments - requires authentication",
    dependencies=[Depends(require_token)]
)
async def list_assessments(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum records to return"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level"),
    clinician_id: Optional[str] = Query(None, description="Filter by clinician"),
    db: Session = Depends(get_db),
    token_data: TokenData = Depends(require_token)
) -> List[AssessmentInDB]:
    """
    Get list of assessments with pagination and filtering.
    
    This is a PROTECTED endpoint - requires authentication.
    
    Args:
        skip: Offset for pagination
        limit: Maximum number of records
        risk_level: Optional filter by risk level
        clinician_id: Optional filter by clinician
        db: Database session
        token_data: Authentication token
        
    Returns:
        List of assessments
    """
    try:
        query = db.query(Assessment)
        
        # Apply filters
        filters = []
        if risk_level:
            filters.append(Assessment.risk_level == risk_level.lower())
        if clinician_id:
            filters.append(Assessment.clinician_id == clinician_id)
        
        if filters:
            query = query.filter(and_(*filters))
        
        # Order by date descending
        assessments = query.order_by(desc(Assessment.assessment_date))\
                          .offset(skip)\
                          .limit(limit)\
                          .all()
        
        return assessments
        
    except Exception as e:
        logger.error(f"Error listing assessments: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving assessments"
        )


@router.get(
    "/assessments/{assessment_id}",
    response_model=AssessmentDetail,
    summary="Get assessment details (Protected)",
    description="Get detailed assessment information - requires authentication",
    dependencies=[Depends(require_token)]
)
async def get_assessment(
    assessment_id: int,
    db: Session = Depends(get_db),
    token_data: TokenData = Depends(require_token)
) -> AssessmentDetail:
    """
    Get detailed information for a specific assessment.
    
    This is a PROTECTED endpoint - requires authentication.
    
    Args:
        assessment_id: Assessment ID
        db: Database session
        token_data: Authentication token
        
    Returns:
        Detailed assessment information
    """
    try:
        assessment = db.query(Assessment).filter(
            Assessment.id == assessment_id
        ).first()
        
        if not assessment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Assessment {assessment_id} not found"
            )
        
        # Get fresh prediction details for recommendations
        predictor = get_predictor()
        assessment_dict = assessment.to_dict()
        
        # Extract feature data
        feature_data = {
            **assessment_dict["demographics"],
            **assessment_dict["clinical_factors"],
            **assessment_dict["sociodemographic_factors"]
        }
        
        # Get risk factors and recommendations
        from app.utils.risk_calculator import (
            calculate_risk_factors,
            generate_recommendations
        )
        
        risk_factors, protective_factors = calculate_risk_factors(
            feature_data,
            assessment.feature_contributions
        )
        
        recommendations = generate_recommendations(
            assessment.risk_level,
            risk_factors,
            feature_data
        )
        
        # Create detailed response
        detail = AssessmentDetail.from_orm(assessment)
        detail.risk_factors = risk_factors
        detail.protective_factors = protective_factors
        detail.recommendations = recommendations
        
        return detail
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting assessment {assessment_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving assessment details"
        )


@router.put(
    "/assessments/{assessment_id}",
    response_model=AssessmentInDB,
    summary="Update assessment (Protected)",
    description="Update assessment data - requires write permission",
    dependencies=[Depends(require_write_permission)]
)
async def update_assessment(
    assessment_id: int,
    update_data: AssessmentUpdate,
    db: Session = Depends(get_db),
    token_data: TokenData = Depends(require_write_permission)
) -> AssessmentInDB:
    """
    Update an existing assessment.
    
    This is a PROTECTED endpoint - requires write permission.
    
    Args:
        assessment_id: Assessment ID
        update_data: Fields to update
        db: Database session
        token_data: Authentication token
        
    Returns:
        Updated assessment
    """
    try:
        assessment = db.query(Assessment).filter(
            Assessment.id == assessment_id
        ).first()
        
        if not assessment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Assessment {assessment_id} not found"
            )
        
        # Update only provided fields
        update_dict = update_data.model_dump(exclude_unset=True)
        
        # If clinical data changed, recalculate risk
        clinical_fields = {
            'age', 'gender', 'consanguinity', 'family_neuro_history',
            'seizures_history', 'brain_injury_history', 'psychiatric_diagnosis',
            'substance_use', 'birth_complications', 'extreme_poverty',
            'education_access_issues', 'healthcare_access', 'disability_diagnosis',
            'social_support_level', 'breastfed_infancy', 'violence_exposure',
            'suicide_ideation', 'psychotropic_medication'
        }
        
        if any(field in update_dict for field in clinical_fields):
            # Update assessment fields first
            for field, value in update_dict.items():
                setattr(assessment, field, value)
            
            # Get updated feature data
            feature_data = {
                "age": assessment.age,
                "gender": assessment.gender,
                "consanguinity": assessment.consanguinity,
                "family_neuro_history": assessment.family_neuro_history,
                "seizures_history": assessment.seizures_history,
                "brain_injury_history": assessment.brain_injury_history,
                "psychiatric_diagnosis": assessment.psychiatric_diagnosis,
                "substance_use": assessment.substance_use,
                "suicide_ideation": assessment.suicide_ideation,
                "psychotropic_medication": assessment.psychotropic_medication,
                "birth_complications": assessment.birth_complications,
                "extreme_poverty": assessment.extreme_poverty,
                "education_access_issues": assessment.education_access_issues,
                "healthcare_access": assessment.healthcare_access,
                "disability_diagnosis": assessment.disability_diagnosis,
                "social_support_level": assessment.social_support_level,
                "breastfed_infancy": assessment.breastfed_infancy,
                "violence_exposure": assessment.violence_exposure
            }
            
            # Recalculate risk
            predictor = get_predictor()
            prediction_result = predictor.predict(feature_data)
            
            # Update risk fields
            assessment.risk_score = prediction_result["risk_score"]
            assessment.risk_level = prediction_result["risk_level"]
            assessment.confidence_score = prediction_result["confidence_score"]
            assessment.feature_contributions = prediction_result.get("feature_importance")
            assessment.model_version = prediction_result["model_version"]
        else:
            # Update non-clinical fields only
            for field, value in update_dict.items():
                setattr(assessment, field, value)
        
        db.commit()
        db.refresh(assessment)
        
        logger.info(f"Assessment {assessment_id} updated by {token_data.username}")
        return assessment
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating assessment {assessment_id}: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error updating assessment"
        )