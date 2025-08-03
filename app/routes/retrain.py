"""
API endpoints for model retraining and monitoring.
Protected endpoints requiring admin authentication.
"""

from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, UploadFile, File
from pydantic import BaseModel, Field

from app.auth import require_admin, TokenData
from app.config import settings
from app.database import SessionLocal
from app.models.assessment import Assessment
from scripts.retrain_with_real_data import IncrementalModelTrainer
from scripts.preprocess_real_data import RealDataPreprocessor
from scripts.evaluate_model import ModelEvaluator
from loguru import logger

router = APIRouter(
    prefix="/api/v1/retrain",
    tags=["retraining"],
    dependencies=[Depends(require_admin)],
    responses={401: {"description": "Unauthorized"}, 403: {"description": "Admin access required"}},
)


class RetrainingRequest(BaseModel):
    """Request model for manual retraining."""

    force: bool = Field(default=False, description="Force retraining even if conditions aren't met")
    min_samples: int = Field(default=50, ge=10, description="Minimum samples required")
    include_synthetic: bool = Field(default=True, description="Include synthetic data in training")


class RetrainingResponse(BaseModel):
    """Response model for retraining operations."""

    status: str
    message: str
    task_id: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    data_composition: Optional[Dict[str, int]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Global task storage (in production, use Redis or similar)
retraining_tasks = {}


def run_retraining_task(task_id: str, request: RetrainingRequest):
    """
    Background task for model retraining.

    Args:
        task_id: Unique task identifier
        request: Retraining parameters
    """
    try:
        # Update task status
        retraining_tasks[task_id] = {
            "status": "running",
            "started_at": datetime.utcnow(),
            "progress": 0,
        }

        # Initialize trainer
        trainer = IncrementalModelTrainer(min_new_samples=request.min_samples)

        # Run retraining
        result = trainer.run_retraining_pipeline(force=request.force)

        # Update task with results
        retraining_tasks[task_id].update(
            {
                "status": "completed",
                "completed_at": datetime.utcnow(),
                "result": result,
                "progress": 100,
            }
        )

        logger.info(f"Retraining task {task_id} completed successfully")

    except Exception as e:
        logger.error(f"Retraining task {task_id} failed: {str(e)}")
        retraining_tasks[task_id] = {
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.utcnow(),
            "progress": -1,
        }


@router.post(
    "/start",
    response_model=RetrainingResponse,
    summary="Start model retraining",
    description="Initiate model retraining process (admin only)",
)
async def start_retraining(
    request: RetrainingRequest,
    background_tasks: BackgroundTasks,
    token_data: TokenData = Depends(require_admin),
) -> RetrainingResponse:
    """
    Start model retraining in the background.

    Args:
        request: Retraining parameters
        background_tasks: FastAPI background tasks
        token_data: Admin authentication

    Returns:
        Task information
    """
    # Generate task ID
    task_id = f"retrain_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    # Check if another retraining is in progress
    active_tasks = [
        tid for tid, task in retraining_tasks.items() if task.get("status") == "running"
    ]

    if active_tasks and not request.force:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Another retraining task is already in progress",
        )

    # Start background task
    background_tasks.add_task(run_retraining_task, task_id, request)

    logger.info(f"Started retraining task {task_id} by user {token_data.username}")

    return RetrainingResponse(
        status="started", message="Retraining task started successfully", task_id=task_id
    )


@router.get(
    "/status/{task_id}",
    summary="Check retraining status",
    description="Get status of a retraining task",
)
async def get_retraining_status(
    task_id: str, token_data: TokenData = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Get status of a retraining task.

    Args:
        task_id: Task identifier
        token_data: Admin authentication

    Returns:
        Task status and results
    """
    if task_id not in retraining_tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Task {task_id} not found"
        )

    task_info = retraining_tasks[task_id].copy()

    # Add duration if completed
    if "started_at" in task_info and "completed_at" in task_info:
        duration = (task_info["completed_at"] - task_info["started_at"]).total_seconds()
        task_info["duration_seconds"] = duration

    return task_info


@router.get(
    "/history", summary="Get retraining history", description="Get history of retraining tasks"
)
async def get_retraining_history(
    limit: int = 10, token_data: TokenData = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Get retraining history.

    Args:
        limit: Maximum number of tasks to return
        token_data: Admin authentication

    Returns:
        List of recent retraining tasks
    """
    # Sort tasks by completion time
    sorted_tasks = sorted(
        [(tid, task) for tid, task in retraining_tasks.items()],
        key=lambda x: x[1].get("completed_at", datetime.min),
        reverse=True,
    )

    # Return limited history
    history = []
    for task_id, task_info in sorted_tasks[:limit]:
        summary = {
            "task_id": task_id,
            "status": task_info.get("status"),
            "started_at": task_info.get("started_at"),
            "completed_at": task_info.get("completed_at"),
            "progress": task_info.get("progress", 0),
        }

        if "result" in task_info:
            summary["result_status"] = task_info["result"].get("status")
            if "metrics" in task_info["result"]:
                summary["metrics"] = {
                    k: v
                    for k, v in task_info["result"]["metrics"].items()
                    if isinstance(v, (int, float))
                }

        history.append(summary)

    return {"total_tasks": len(retraining_tasks), "history": history}


@router.post(
    "/upload-data",
    summary="Upload clinical data for retraining",
    description="Upload new clinical data file for preprocessing and retraining",
)
async def upload_clinical_data(
    file: UploadFile = File(...),
    process_immediately: bool = True,
    token_data: TokenData = Depends(require_admin),
) -> Dict[str, Any]:
    """
    Upload clinical data for retraining.

    Args:
        file: CSV or Excel file with clinical data
        process_immediately: Process file immediately
        token_data: Admin authentication

    Returns:
        Upload and processing status
    """
    # Validate file type
    if not file.filename.endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="File must be CSV or Excel format"
        )

    # Save uploaded file
    upload_dir = settings.synthetic_data_dir / "uploads"
    upload_dir.mkdir(exist_ok=True)

    file_path = upload_dir / f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{file.filename}"

    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)

        logger.info(f"Saved uploaded file to {file_path}")

        if process_immediately:
            # Process the file
            preprocessor = RealDataPreprocessor(str(file_path))
            analysis = preprocessor.analyze_columns()

            # Check if we have enough valid data
            if len(analysis["missing_features"]) > 10:
                return {
                    "status": "warning",
                    "message": "File uploaded but missing many required features",
                    "file_path": str(file_path),
                    "analysis": analysis,
                }

            # Preprocess and save
            preprocessor.preprocess_data()
            processed_path = preprocessor.save_processed_data()

            return {
                "status": "success",
                "message": "File uploaded and processed successfully",
                "file_path": str(file_path),
                "processed_path": str(processed_path),
                "analysis": analysis,
            }
        else:
            return {
                "status": "success",
                "message": "File uploaded successfully",
                "file_path": str(file_path),
            }

    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}",
        )


@router.get(
    "/metrics/comparison",
    summary="Compare model versions",
    description="Compare performance metrics across model versions",
)
async def compare_model_versions(
    versions: int = 5, token_data: TokenData = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Compare recent model versions.

    Args:
        versions: Number of versions to compare
        token_data: Admin authentication

    Returns:
        Model comparison data
    """
    try:
        # Find model files
        model_files = sorted(
            settings.models_dir.glob("model_v*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True
        )[:versions]

        if not model_files:
            return {"status": "no_models", "message": "No model versions found"}

        # Compare models
        evaluator = ModelEvaluator()
        comparison_df = evaluator.compare_models(model_files)

        # Convert to dict for JSON response
        comparison_data = comparison_df.to_dict("records")

        # Add trends
        if len(comparison_data) > 1:
            trends = {}
            for metric in ["accuracy", "precision", "recall", "f1_score", "auc_roc"]:
                values = [row[metric] for row in comparison_data if metric in row]
                if len(values) > 1:
                    trend = "improving" if values[0] > values[-1] else "declining"
                    change = values[0] - values[-1]
                    trends[metric] = {
                        "trend": trend,
                        "change": change,
                        "current": values[0],
                        "previous": values[-1],
                    }

            return {
                "status": "success",
                "versions_compared": len(comparison_data),
                "comparison": comparison_data,
                "trends": trends,
            }

        return {
            "status": "success",
            "versions_compared": len(comparison_data),
            "comparison": comparison_data,
        }

    except Exception as e:
        logger.error(f"Error comparing models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error comparing models: {str(e)}",
        )
