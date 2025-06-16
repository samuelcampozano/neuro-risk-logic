"""
NeuroRiskLogic - AI-Powered Neurodevelopmental Risk Assessment System

A machine learning system that evaluates neurodevelopmental risk factors
using clinical and sociodemographic features without requiring explicit diagnoses.

Author: Samuel Campozano Lopez
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Samuel Campozano Lopez"
__email__ = "samuelco860@gmail.com"

# Make key components easily importable
from app.config import settings
from app.database import engine, SessionLocal, get_db

__all__ = ["settings", "engine", "SessionLocal", "get_db", "__version__"]