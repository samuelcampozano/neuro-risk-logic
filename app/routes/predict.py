"""
Prediction endpoint for neurodevelopmental risk assessment.
Public endpoint - no authentication required.
"""

from fastapi import APIRouter, HTTPException, status
from app.schemas.request import PredictionRequest
from app.schemas.response import PredictionResponse, ErrorResponse
from app.models.predictor import get_predictor
from loguru import logger

router = APIRouter(
    prefix="/api/v1",
    tags=["predictions"],
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Make risk prediction",
    description="Perform neurodevelopmental risk assessment without storing data",
    responses={
        200: {"description": "Successful prediction", "model": PredictionResponse},
        400: {"description": "Invalid input data"},
        503: {"description": "Model not available"},
    },
)
async def predict_risk(request: PredictionRequest) -> PredictionResponse:
    """
    Make a neurodevelopmental risk prediction.

    This endpoint:
    - Validates input features
    - Runs ML model prediction
    - Returns risk score, factors, and recommendations
    - Does NOT store any data

    Args:
        request: Prediction request with 18 features

    Returns:
        PredictionResponse with risk assessment
    """
    try:
        # Get predictor instance
        predictor = get_predictor()

        # Check if model is loaded
        if not predictor.is_loaded:
            logger.error("Model not loaded")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction model is not available. Please try again later.",
            )

        # Convert request to dict for prediction
        assessment_data = request.model_dump()

        # Make prediction
        result = predictor.predict(assessment_data)

        # Create response
        response = PredictionResponse(
            risk_score=result["risk_score"],
            risk_level=result["risk_level"],
            confidence_score=result["confidence_score"],
            risk_factors=result["risk_factors"],
            protective_factors=result["protective_factors"],
            feature_importance=result.get("feature_importance"),
            recommendations=result["recommendations"],
        )

        logger.info(
            f"Prediction completed - Risk: {result['risk_level']} " f"({result['risk_score']:.2f})"
        )

        return response

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during prediction",
        )


@router.get(
    "/model/info",
    summary="Get model information",
    description="Get information about the current ML model",
    tags=["model"],
)
async def get_model_info():
    """
    Get information about the loaded model.

    Returns:
        Model metadata and configuration
    """
    try:
        predictor = get_predictor()
        return predictor.get_model_info()
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving model information",
        )
