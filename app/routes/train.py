"""
Training endpoints for manual model training operations.
Protected endpoints requiring admin authentication.
"""

from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field

from app.auth import require_admin, TokenData
from app.config import settings
from scripts.train_model import ModelTrainer
from scripts.generate_synthetic_data import SyntheticDataGenerator
from loguru import logger

router = APIRouter(
    prefix="/api/v1/train",
    tags=["training"],
    dependencies=[Depends(require_admin)],
    responses={
        401: {"description": "Unauthorized"},
        403: {"description": "Admin access required"}
    }
)


class TrainingRequest(BaseModel):
    """Request model for model training."""
    n_samples: int = Field(
        default=1000,
        ge=100,
        description="Number of synthetic samples to generate"
    )
    test_size: float = Field(
        default=0.2,
        ge=0.1,
        le=0.5,
        description="Proportion of data for testing"
    )
    create_visualizations: bool = Field(
        default=True,
        description="Generate visualization plots"
    )


class TrainingResponse(BaseModel):
    """Response model for training operations."""
    status: str
    message: str
    task_id: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    model_version: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Global task storage (in production, use Redis or similar)
training_tasks = {}


def run_training_task(task_id: str, request: TrainingRequest):
    """
    Background task for model training.
    
    Args:
        task_id: Unique task identifier
        request: Training parameters
    """
    try:
        # Update task status
        training_tasks[task_id] = {
            "status": "running",
            "started_at": datetime.utcnow(),
            "progress": 0,
            "stage": "generating_data"
        }
        
        # Step 1: Generate synthetic data
        logger.info(f"Generating {request.n_samples} synthetic samples...")
        generator = SyntheticDataGenerator(
            n_samples=request.n_samples,
            random_seed=settings.random_seed
        )
        
        df = generator.generate_dataset()
        output_file = generator.save_dataset()
        
        training_tasks[task_id]["progress"] = 30
        training_tasks[task_id]["stage"] = "preparing_training"
        
        # Step 2: Train model
        logger.info("Starting model training...")
        trainer = ModelTrainer()
        
        # Load data from the generated file
        X, y = trainer.load_data_from_csv(output_file)
        
        training_tasks[task_id]["progress"] = 50
        training_tasks[task_id]["stage"] = "training_model"
        
        # Train the model
        metrics = trainer.train_model(X, y)
        
        training_tasks[task_id]["progress"] = 80
        training_tasks[task_id]["stage"] = "saving_model"
        
        # Save model
        trainer.save_model()
        
        # Create visualizations if requested
        if request.create_visualizations:
            training_tasks[task_id]["stage"] = "creating_visualizations"
            trainer.create_visualizations()
        
        training_tasks[task_id]["progress"] = 100
        
        # Update task with results
        training_tasks[task_id].update({
            "status": "completed",
            "completed_at": datetime.utcnow(),
            "metrics": metrics,
            "model_version": trainer.model_version,
            "data_file": str(output_file),
            "progress": 100
        })
        
        logger.info(f"Training task {task_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Training task {task_id} failed: {str(e)}")
        training_tasks[task_id] = {
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.utcnow(),
            "progress": -1
        }


@router.post(
    "/start",
    response_model=TrainingResponse,
    summary="Start model training",
    description="Train a new model from synthetic data (admin only)"
)
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    token_data: TokenData = Depends(require_admin)
) -> TrainingResponse:
    """
    Start model training in the background.
    
    Args:
        request: Training parameters
        background_tasks: FastAPI background tasks
        token_data: Admin authentication
        
    Returns:
        Task information
    """
    # Generate task ID
    task_id = f"train_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    # Check if another training is in progress
    active_tasks = [
        tid for tid, task in training_tasks.items()
        if task.get("status") == "running"
    ]
    
    if active_tasks:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Another training task is already in progress"
        )
    
    # Start background task
    background_tasks.add_task(run_training_task, task_id, request)
    
    logger.info(f"Started training task {task_id} by user {token_data.username}")
    
    return TrainingResponse(
        status="started",
        message="Training task started successfully",
        task_id=task_id
    )


@router.get(
    "/status/{task_id}",
    summary="Check training status",
    description="Get status of a training task"
)
async def get_training_status(
    task_id: str,
    token_data: TokenData = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Get status of a training task.
    
    Args:
        task_id: Task identifier
        token_data: Admin authentication
        
    Returns:
        Task status and results
    """
    if task_id not in training_tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    
    task_info = training_tasks[task_id].copy()
    
    # Add duration if completed
    if "started_at" in task_info and "completed_at" in task_info:
        duration = (task_info["completed_at"] - task_info["started_at"]).total_seconds()
        task_info["duration_seconds"] = duration
    
    return task_info


@router.get(
    "/history",
    summary="Get training history",
    description="Get history of training tasks"
)
async def get_training_history(
    limit: int = 10,
    token_data: TokenData = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Get training history.
    
    Args:
        limit: Maximum number of tasks to return
        token_data: Admin authentication
        
    Returns:
        List of recent training tasks
    """
    # Sort tasks by completion time
    sorted_tasks = sorted(
        [(tid, task) for tid, task in training_tasks.items()],
        key=lambda x: x[1].get("completed_at", datetime.min),
        reverse=True
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
            "stage": task_info.get("stage", "unknown")
        }
        
        if task_info.get("status") == "completed":
            summary["model_version"] = task_info.get("model_version")
            if "metrics" in task_info:
                summary["metrics"] = {
                    k: v for k, v in task_info["metrics"].items()
                    if isinstance(v, (int, float))
                }
        elif task_info.get("status") == "failed":
            summary["error"] = task_info.get("error")
        
        history.append(summary)
    
    return {
        "total_tasks": len(training_tasks),
        "history": history
    }


@router.delete(
    "/history",
    summary="Clear training history",
    description="Clear all completed training tasks from history"
)
async def clear_training_history(
    token_data: TokenData = Depends(require_admin)
) -> Dict[str, str]:
    """
    Clear training history (keep only running tasks).
    
    Args:
        token_data: Admin authentication
        
    Returns:
        Confirmation message
    """
    # Remove completed and failed tasks
    tasks_to_remove = [
        tid for tid, task in training_tasks.items()
        if task.get("status") in ["completed", "failed"]
    ]
    
    for tid in tasks_to_remove:
        del training_tasks[tid]
    
    logger.info(f"Cleared {len(tasks_to_remove)} training tasks from history")
    
    return {
        "message": f"Cleared {len(tasks_to_remove)} tasks from history"
    }