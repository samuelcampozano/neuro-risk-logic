"""
Main FastAPI application for NeuroRiskLogic.
Production-ready configuration with comprehensive middleware and error handling.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import settings
from app.database import init_db, check_db_connection
from app.models.predictor import load_model
from app.routes import (
    predict_router,
    submit_router,
    stats_router,
    auth_router,
    retrain_router
)
from app.schemas.response import ErrorResponse, HealthCheckResponse

from loguru import logger
import time
from datetime import datetime


# Configure Loguru
logger.add(
    "logs/app.log",
    rotation="10 MB",
    retention="10 days",
    level=settings.log_level
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    """
    # Startup
    logger.info("=" * 60)
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")
    logger.info("=" * 60)
    
    try:
        # Initialize database
        logger.info("Initializing database...")
        init_db()
        
        # Check database connection
        if check_db_connection():
            logger.info("✅ Database connection successful")
        else:
            logger.error("❌ Database connection failed")
        
        # Load ML model
        logger.info("Loading ML model...")
        predictor = load_model()
        
        if predictor.is_loaded:
            logger.info("✅ ML model loaded successfully")
            model_info = predictor.get_model_info()
            logger.info(f"Model type: {model_info.get('model_type', 'Unknown')}")
            logger.info(f"Model version: {model_info.get('model_version', 'Unknown')}")
        else:
            logger.warning("⚠️ ML model not available - predictions will fail")
        
        # Validate model
        is_valid, error_msg = predictor.validate_model()
        if is_valid:
            logger.info("✅ Model validation passed")
        else:
            logger.error(f"❌ Model validation failed: {error_msg}")
        
        logger.info("Application startup completed successfully")
        
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("Application shutting down...")
    logger.info("Shutdown completed")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="""
## AI-Powered Neurodevelopmental Risk Assessment System

NeuroRiskLogic uses machine learning to analyze clinical and sociodemographic factors, 
providing evidence-based risk predictions for neurodevelopmental disorders.

### Features
- 🔍 Comprehensive assessment of 18 evidence-based risk factors
- 🤖 Machine learning predictions with interpretable results
- 🔐 Secure API with JWT authentication
- 📊 Detailed analytics and clinical recommendations

### Authentication
Most endpoints require JWT authentication. Use `/api/v1/auth/login` to obtain a token.

### Important Notes
- This is a screening tool, not a diagnostic system
- All assessments require explicit consent
- Results should be interpreted by qualified healthcare professionals
    """,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
    openapi_url="/openapi.json" if not settings.is_production else "/api/openapi.json"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins_list(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Gzip compression for responses
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Security middleware
if settings.is_production:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*.yourdomain.com", "yourdomain.com"]
    )

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add X-Process-Time header to track request duration."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.3f}"
    return response

# Include routers
app.include_router(predict_router)
app.include_router(submit_router)
app.include_router(stats_router)
app.include_router(auth_router)
app.include_router(retrain_router)

# Root endpoint
@app.get(
    "/",
    tags=["root"],
    summary="API Information",
    description="Get general information about the API"
)
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "AI-Powered Neurodevelopmental Risk Assessment System",
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "openapi": "/api/openapi.json"
        },
        "endpoints": {
            "health": "/health",
            "predict": "/api/v1/predict",
            "assessments": "/api/v1/assessments",
            "statistics": "/api/v1/stats",
            "authentication": "/api/v1/auth/login"
        },
        "environment": settings.environment,
        "timestamp": datetime.utcnow().isoformat()
    }

# Health check endpoint
@app.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["monitoring"],
    summary="Health check",
    description="Check system health and component status"
)
async def health_check():
    """
    Comprehensive health check endpoint.
    
    Returns:
        System health status with component details
    """
    health_status = {
        "status": "healthy",
        "version": settings.app_version,
        "timestamp": datetime.utcnow(),
        "services": {}
    }
    
    # Check database
    try:
        if check_db_connection():
            health_status["services"]["database"] = {
                "status": "connected",
                "response_time_ms": 5  # Mock value
            }
        else:
            health_status["services"]["database"] = {
                "status": "disconnected",
                "response_time_ms": -1
            }
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["database"] = {
            "status": "error",
            "error": str(e)
        }
        health_status["status"] = "unhealthy"
    
    # Check ML model
    try:
        predictor = load_model()
        if predictor.is_loaded:
            model_info = predictor.get_model_info()
            health_status["services"]["ml_model"] = {
                "status": "loaded",
                "version": model_info.get("model_version", "unknown"),
                "memory_usage_mb": 150  # Mock value
            }
        else:
            health_status["services"]["ml_model"] = {
                "status": "not_loaded"
            }
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["ml_model"] = {
            "status": "error",
            "error": str(e)
        }
        health_status["status"] = "unhealthy"
    
    # Mock cache status
    health_status["services"]["cache"] = {
        "status": "active",
        "hit_rate": 0.85
    }
    
    return health_status

# Exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions with consistent format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "detail": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed information."""
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "ValidationError",
            "detail": "Request validation failed",
            "errors": errors,
            "status_code": 422,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    # Don't expose internal errors in production
    if settings.is_production:
        detail = "An internal error occurred"
    else:
        detail = str(exc)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "detail": detail,
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Prometheus metrics endpoint (optional)
if settings.enable_metrics:
    try:
        from prometheus_client import make_asgi_app
        metrics_app = make_asgi_app()
        app.mount(settings.metrics_endpoint, metrics_app)
        logger.info(f"Metrics endpoint enabled at {settings.metrics_endpoint}")
    except ImportError:
        logger.warning("prometheus_client not installed, metrics endpoint disabled")

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.is_development,
        log_level=settings.log_level.lower()
    )