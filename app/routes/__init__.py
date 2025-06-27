"""
API route handlers for NeuroRiskLogic system.
"""

from app.routes.predict import router as predict_router
from app.routes.submit import router as submit_router
from app.routes.stats import router as stats_router
from app.routes.auth import router as auth_router
from app.routes.train import router as train_router
from app.routes.retrain import router as retrain_router

__all__ = [
    "predict_router",
    "submit_router", 
    "stats_router",
    "auth_router",
    "train_router",
    "retrain_router"
]