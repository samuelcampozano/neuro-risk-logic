"""
Configuration management using Pydantic Settings.
Handles environment variables and application configuration.
"""

from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator
import os
from pathlib import Path

# Determine project root directory
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    
    # Application Info
    app_name: str = Field(default="NeuroRiskLogic", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENV")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # API Configuration
    api_prefix: str = Field(default="/api/v1", env="API_PREFIX")
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    
    # Database
    database_url: str = Field(
        default="postgresql://user:password@localhost:5432/neurorisk_db",
        env="DATABASE_URL"
    )
    postgres_user: Optional[str] = Field(default=None, env="POSTGRES_USER")
    postgres_password: Optional[str] = Field(default=None, env="POSTGRES_PASSWORD")
    postgres_db: Optional[str] = Field(default=None, env="POSTGRES_DB")
    postgres_host: Optional[str] = Field(default=None, env="POSTGRES_HOST")
    postgres_port: Optional[int] = Field(default=None, env="POSTGRES_PORT")
    
    # Security
    secret_key: str = Field(
        default="your-secret-key-change-this-in-production",
        env="SECRET_KEY"
    )
    api_key: str = Field(
        default="your-api-key-for-admin-endpoints",
        env="API_KEY"
    )
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        env="CORS_ORIGINS"
    )
    
    # Model Configuration
    model_version: str = Field(default="v1.0.0", env="MODEL_VERSION")
    min_training_samples: int = Field(default=100, env="MIN_TRAINING_SAMPLES")
    retrain_threshold_accuracy: float = Field(default=0.75, env="RETRAIN_THRESHOLD_ACCURACY")
    
    # Paths
    feature_definitions_path: str = Field(
        default="data/feature_definitions.json",
        env="FEATURE_DEFINITIONS_PATH"
    )
    models_dir: Path = Field(default=BASE_DIR / "data" / "models")
    synthetic_data_dir: Path = Field(default=BASE_DIR / "data" / "synthetic")
    
    # Synthetic Data Generation
    synthetic_samples: int = Field(default=1000, env="SYNTHETIC_SAMPLES")
    random_seed: int = Field(default=42, env="RANDOM_SEED")
    
    # Monitoring
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_endpoint: str = Field(default="/metrics", env="METRICS_ENDPOINT")
    
    # External Services (Optional)
    sentry_dsn: Optional[str] = Field(default=None, env="SENTRY_DSN")
    slack_webhook_url: Optional[str] = Field(default=None, env="SLACK_WEBHOOK_URL")
    
    # ML Tracking (Optional)
    mlflow_tracking_uri: Optional[str] = Field(default=None, env="MLFLOW_TRACKING_URI")
    wandb_api_key: Optional[str] = Field(default=None, env="WANDB_API_KEY")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    @validator("cors_origins", pre=True)
    def assemble_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("database_url", pre=True)
    def assemble_db_connection(cls, v, values):
        """Construct database URL from components if not directly provided."""
        if v and v != "postgresql://user:password@localhost:5432/neurorisk_db":
            return v
        
        # Try to construct from components
        user = values.get("postgres_user")
        password = values.get("postgres_password")
        host = values.get("postgres_host", "localhost")
        port = values.get("postgres_port", 5432)
        db = values.get("postgres_db", "neurorisk_db")
        
        if user and password:
            return f"postgresql://{user}:{password}@{host}:{port}/{db}"
        
        # Fallback to SQLite for development
        return f"sqlite:///{BASE_DIR}/data/neurorisk.db"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment.lower() == "development"
    
    def get_model_path(self, version: Optional[str] = None) -> Path:
        """Get path to model file."""
        if version:
            return self.models_dir / f"model_{version}.pkl"
        return self.models_dir / "model_current.pkl"
    
    def get_feature_definitions_path(self) -> Path:
        """Get full path to feature definitions file."""
        return BASE_DIR / self.feature_definitions_path


# Create global settings instance
settings = Settings()

# Create necessary directories
settings.models_dir.mkdir(parents=True, exist_ok=True)
settings.synthetic_data_dir.mkdir(parents=True, exist_ok=True)