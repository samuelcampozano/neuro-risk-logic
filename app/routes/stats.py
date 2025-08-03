"""
Statistics and analytics endpoints.
Protected endpoints requiring authentication.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from app.database import get_db
from app.models.assessment import Assessment
from app.schemas.response import StatsResponse
from app.auth import require_token, TokenData
from loguru import logger

router = APIRouter(
    prefix="/api/v1",
    tags=["statistics"],
    dependencies=[Depends(require_token)],
    responses={401: {"description": "Unauthorized"}},
)


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Get system statistics",
    description="Get comprehensive statistics about assessments",
)
async def get_statistics(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    db: Session = Depends(get_db),
    token_data: TokenData = Depends(require_token),
) -> StatsResponse:
    """
    Get comprehensive system statistics.

    Args:
        days: Number of days to include in recent stats
        db: Database session
        token_data: Authentication token

    Returns:
        Comprehensive statistics
    """
    try:
        # Total assessments
        total_assessments = db.query(Assessment).count()

        # Assessments by risk level
        risk_level_counts = (
            db.query(Assessment.risk_level, func.count(Assessment.id))
            .group_by(Assessment.risk_level)
            .all()
        )

        assessments_by_risk_level = {level: count for level, count in risk_level_counts}

        # Ensure all levels are represented
        for level in ["low", "moderate", "high"]:
            if level not in assessments_by_risk_level:
                assessments_by_risk_level[level] = 0

        # Assessments by gender
        gender_counts = (
            db.query(Assessment.gender, func.count(Assessment.id)).group_by(Assessment.gender).all()
        )

        assessments_by_gender = {gender: count for gender, count in gender_counts}

        # Average risk score
        avg_risk = db.query(func.avg(Assessment.risk_score)).scalar() or 0.0

        # Average age
        avg_age = db.query(func.avg(Assessment.age)).scalar() or 0.0

        # Most common risk factors (high-risk assessments)
        high_risk_assessments = db.query(Assessment).filter(Assessment.risk_level == "high").all()

        risk_factor_frequency = {}
        for assessment in high_risk_assessments:
            # Count clinical risk factors
            if assessment.consanguinity:
                risk_factor_frequency["consanguinity"] = (
                    risk_factor_frequency.get("consanguinity", 0) + 1
                )
            if assessment.family_neuro_history:
                risk_factor_frequency["family_neuro_history"] = (
                    risk_factor_frequency.get("family_neuro_history", 0) + 1
                )
            if assessment.seizures_history:
                risk_factor_frequency["seizures_history"] = (
                    risk_factor_frequency.get("seizures_history", 0) + 1
                )
            if assessment.brain_injury_history:
                risk_factor_frequency["brain_injury_history"] = (
                    risk_factor_frequency.get("brain_injury_history", 0) + 1
                )
            if assessment.psychiatric_diagnosis:
                risk_factor_frequency["psychiatric_diagnosis"] = (
                    risk_factor_frequency.get("psychiatric_diagnosis", 0) + 1
                )
            if assessment.substance_use:
                risk_factor_frequency["substance_use"] = (
                    risk_factor_frequency.get("substance_use", 0) + 1
                )
            if assessment.suicide_ideation:
                risk_factor_frequency["suicide_ideation"] = (
                    risk_factor_frequency.get("suicide_ideation", 0) + 1
                )
            if assessment.extreme_poverty:
                risk_factor_frequency["extreme_poverty"] = (
                    risk_factor_frequency.get("extreme_poverty", 0) + 1
                )
            if assessment.violence_exposure:
                risk_factor_frequency["violence_exposure"] = (
                    risk_factor_frequency.get("violence_exposure", 0) + 1
                )

        # Calculate frequencies
        total_high_risk = len(high_risk_assessments)
        most_common_risk_factors = []

        if total_high_risk > 0:
            for factor, count in sorted(
                risk_factor_frequency.items(), key=lambda x: x[1], reverse=True
            )[:5]:
                frequency = count / total_high_risk
                most_common_risk_factors.append({"factor": factor, "frequency": frequency})

        # Recent assessments
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_count = (
            db.query(Assessment).filter(Assessment.assessment_date >= cutoff_date).count()
        )

        # Model performance (mock data - would come from model metrics)
        model_performance = {"current_accuracy": 0.86, "current_auc": 0.92}

        return StatsResponse(
            total_assessments=total_assessments,
            assessments_by_risk_level=assessments_by_risk_level,
            assessments_by_gender=assessments_by_gender,
            average_risk_score=float(avg_risk),
            average_age=float(avg_age),
            most_common_risk_factors=most_common_risk_factors,
            assessments_last_30_days=recent_count,
            model_performance=model_performance,
        )

    except Exception as e:
        logger.error(f"Error calculating statistics: {str(e)}")
        raise HTTPException(status_code=500, detail="Error calculating statistics")


@router.get(
    "/stats/trends", summary="Get trend analysis", description="Get assessment trends over time"
)
async def get_trends(
    period: str = Query("month", regex="^(week|month|quarter|year)$"),
    db: Session = Depends(get_db),
    token_data: TokenData = Depends(require_token),
) -> Dict[str, Any]:
    """
    Get assessment trends over specified period.

    Args:
        period: Time period for grouping
        db: Database session
        token_data: Authentication token

    Returns:
        Trend data
    """
    try:
        # Determine date range
        end_date = datetime.utcnow()
        if period == "week":
            start_date = end_date - timedelta(days=7)
            date_format = "%Y-%m-%d"
        elif period == "month":
            start_date = end_date - timedelta(days=30)
            date_format = "%Y-%m-%d"
        elif period == "quarter":
            start_date = end_date - timedelta(days=90)
            date_format = "%Y-%W"  # Week number
        else:  # year
            start_date = end_date - timedelta(days=365)
            date_format = "%Y-%m"  # Month

        # Get assessments in date range
        assessments = db.query(Assessment).filter(Assessment.assessment_date >= start_date).all()

        # Group by date
        trends = {}
        risk_trends = {"low": {}, "moderate": {}, "high": {}}

        for assessment in assessments:
            date_key = assessment.assessment_date.strftime(date_format)

            # Overall count
            trends[date_key] = trends.get(date_key, 0) + 1

            # Risk level trends
            risk_level = assessment.risk_level
            if risk_level not in risk_trends:
                risk_trends[risk_level] = {}
            risk_trends[risk_level][date_key] = risk_trends[risk_level].get(date_key, 0) + 1

        return {
            "period": period,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "total_assessments": len(assessments),
            "daily_counts": trends,
            "risk_level_trends": risk_trends,
        }

    except Exception as e:
        logger.error(f"Error calculating trends: {str(e)}")
        raise HTTPException(status_code=500, detail="Error calculating trends")


@router.get(
    "/stats/risk-factors",
    summary="Get risk factor analysis",
    description="Analyze prevalence and impact of risk factors",
)
async def analyze_risk_factors(
    db: Session = Depends(get_db), token_data: TokenData = Depends(require_token)
) -> Dict[str, Any]:
    """
    Analyze risk factors across all assessments.

    Args:
        db: Database session
        token_data: Authentication token

    Returns:
        Risk factor analysis
    """
    try:
        # Get all assessments
        assessments = db.query(Assessment).all()
        total = len(assessments)

        if total == 0:
            return {"total_assessments": 0, "risk_factors": [], "protective_factors": []}

        # Analyze each risk factor
        risk_factors = [
            ("consanguinity", "Parental Consanguinity"),
            ("family_neuro_history", "Family Neurological History"),
            ("seizures_history", "Seizures/Convulsions"),
            ("brain_injury_history", "Brain Injury"),
            ("psychiatric_diagnosis", "Psychiatric Diagnosis"),
            ("substance_use", "Substance Use"),
            ("suicide_ideation", "Suicide Ideation"),
            ("psychotropic_medication", "Psychotropic Medication"),
            ("birth_complications", "Birth Complications"),
            ("extreme_poverty", "Extreme Poverty"),
            ("education_access_issues", "Education Access Issues"),
            ("disability_diagnosis", "Disability Diagnosis"),
            ("violence_exposure", "Violence Exposure"),
        ]

        protective_factors = [
            ("healthcare_access", "Healthcare Access"),
            ("breastfed_infancy", "Breastfed in Infancy"),
        ]

        risk_factor_stats = []
        for factor_name, display_name in risk_factors:
            # Count prevalence
            count = sum(1 for a in assessments if getattr(a, factor_name))
            prevalence = count / total

            # Calculate average risk score when factor is present vs absent
            with_factor = [a.risk_score for a in assessments if getattr(a, factor_name)]
            without_factor = [a.risk_score for a in assessments if not getattr(a, factor_name)]

            avg_risk_with = sum(with_factor) / len(with_factor) if with_factor else 0
            avg_risk_without = sum(without_factor) / len(without_factor) if without_factor else 0

            risk_factor_stats.append(
                {
                    "factor": factor_name,
                    "display_name": display_name,
                    "prevalence": prevalence,
                    "count": count,
                    "avg_risk_score_with": avg_risk_with,
                    "avg_risk_score_without": avg_risk_without,
                    "risk_increase": avg_risk_with - avg_risk_without,
                }
            )

        # Sort by risk increase
        risk_factor_stats.sort(key=lambda x: x["risk_increase"], reverse=True)

        # Analyze protective factors
        protective_factor_stats = []
        for factor_name, display_name in protective_factors:
            count = sum(1 for a in assessments if getattr(a, factor_name))
            prevalence = count / total

            with_factor = [a.risk_score for a in assessments if getattr(a, factor_name)]
            without_factor = [a.risk_score for a in assessments if not getattr(a, factor_name)]

            avg_risk_with = sum(with_factor) / len(with_factor) if with_factor else 0
            avg_risk_without = sum(without_factor) / len(without_factor) if without_factor else 0

            protective_factor_stats.append(
                {
                    "factor": factor_name,
                    "display_name": display_name,
                    "prevalence": prevalence,
                    "count": count,
                    "avg_risk_score_with": avg_risk_with,
                    "avg_risk_score_without": avg_risk_without,
                    "risk_reduction": avg_risk_without - avg_risk_with,
                }
            )

        # Social support analysis
        social_support_stats = {}
        for level in ["isolated", "moderate", "supported"]:
            level_assessments = [a for a in assessments if a.social_support_level == level]
            if level_assessments:
                avg_risk = sum(a.risk_score for a in level_assessments) / len(level_assessments)
                social_support_stats[level] = {
                    "count": len(level_assessments),
                    "prevalence": len(level_assessments) / total,
                    "avg_risk_score": avg_risk,
                }

        return {
            "total_assessments": total,
            "risk_factors": risk_factor_stats,
            "protective_factors": protective_factor_stats,
            "social_support_analysis": social_support_stats,
        }

    except Exception as e:
        logger.error(f"Error analyzing risk factors: {str(e)}")
        raise HTTPException(status_code=500, detail="Error analyzing risk factors")
