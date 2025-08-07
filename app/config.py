"""
Configuration management using Pydantic Settings.
Handles environment variables and application configuration.
"""

from typing import List, Optional, Union
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
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

    # Database Configuration - PostgreSQL by default
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/neurorisk_db", env="DATABASE_URL"
    )
    postgres_user: Optional[str] = Field(default="postgres", env="POSTGRES_USER")
    postgres_password: Optional[str] = Field(default="postgres", env="POSTGRES_PASSWORD")
    postgres_db: Optional[str] = Field(default="neurorisk_db", env="POSTGRES_DB")
    postgres_host: Optional[str] = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: Optional[int] = Field(default=5432, env="POSTGRES_PORT")

    # Security
    secret_key: str = Field(default="your-secret-key-change-this-in-production", env="SECRET_KEY")
    api_key: str = Field(default="your-api-key-for-admin-endpoints", env="API_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")

    # CORS - Changed to Union type to handle both string and list
    cors_origins: Union[str, List[str]] = Field(
        default="http://localhost:3000,http://localhost:8080", env="CORS_ORIGINS"
    )

    # Model Configuration
    ml_model_version: str = Field(default="v1.0.0", env="MODEL_VERSION")
    min_training_samples: int = Field(default=100, env="MIN_TRAINING_SAMPLES")
    retrain_threshold_accuracy: float = Field(default=0.75, env="RETRAIN_THRESHOLD_ACCURACY")

    # Paths
    feature_definitions_path: str = Field(
        default="data/feature_definitions.json", env="FEATURE_DEFINITIONS_PATH"
    )
    models_dir: Path = Field(default=BASE_DIR / "data" / "models")
    synthetic_data_dir: Path = Field(default=BASE_DIR / "data" / "synthetic")

    # Synthetic Data Generation
    synthetic_samples: int = Field(default=1000, env="SYNTHETIC_SAMPLES")
    random_seed: int = Field(default=42, env="RANDOM_SEED")

    # Incremental Learning
    enable_incremental_learning: bool = Field(default=True, env="ENABLE_INCREMENTAL_LEARNING")
    incremental_retrain_interval: int = Field(default=7, env="INCREMENTAL_RETRAIN_INTERVAL")
    incremental_confidence_threshold: float = Field(
        default=0.85, env="INCREMENTAL_CONFIDENCE_THRESHOLD"
    )

    # Monitoring
    enable_metrics: bool = Field(default=False, env="ENABLE_METRICS")
    metrics_endpoint: str = Field(default="/metrics", env="METRICS_ENDPOINT")

    # External Services (Optional)
    sentry_dsn: Optional[str] = Field(default=None, env="SENTRY_DSN")
    slack_webhook_url: Optional[str] = Field(default=None, env="SLACK_WEBHOOK_URL")

    # ML Tracking (Optional)
    mlflow_tracking_uri: Optional[str] = Field(default=None, env="MLFLOW_TRACKING_URI")
    wandb_api_key: Optional[str] = Field(default=None, env="WANDB_API_KEY")

    # Cloud Storage (Optional)
    aws_access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    aws_bucket_name: Optional[str] = Field(default=None, env="AWS_BUCKET_NAME")

    # Email Notifications (Optional)
    smtp_host: Optional[str] = Field(default=None, env="SMTP_HOST")
    smtp_port: int = Field(default=587, env="SMTP_PORT")
    smtp_user: Optional[str] = Field(default=None, env="SMTP_USER")
    smtp_password: Optional[str] = Field(default=None, env="SMTP_PASSWORD")
    notification_email: Optional[str] = Field(default=None, env="NOTIFICATION_EMAIL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # This prevents the model_ namespace conflict
        protected_namespaces=("settings_",),
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string or return list."""
        if isinstance(v, str):
            # Handle empty string
            if not v.strip():
                return ["http://localhost:3000", "http://localhost:8080"]
            # Handle comma-separated string
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        elif isinstance(v, list):
            return v
        else:
            # Default fallback
            return ["http://localhost:3000", "http://localhost:8080"]

    @field_validator("database_url", mode="before")
    @classmethod
    def assemble_db_connection(cls, v, info):
        """Construct database URL from components if not directly provided."""
        # If DATABASE_URL is explicitly set and valid, use it
        if v and not v.startswith("postgresql://user:password"):
            return v

        # Get values from validation context
        values = info.data
        environment = values.get("environment", "development")

        # Use SQLite for testing and CI environments
        if environment.lower() in ("test", "testing", "ci"):
            return "sqlite:///./data/test_neurorisk.db"

        # Try to construct from components for other environments
        user = values.get("postgres_user", "postgres")
        password = values.get("postgres_password", "postgres")
        host = values.get("postgres_host", "localhost")
        port = values.get("postgres_port", 5432)
        db = values.get("postgres_db", "neurorisk_db")

        # Always construct PostgreSQL URL for production readiness
        return f"postgresql://{user}:{password}@{host}:{port}/{db}"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment.lower() == "development"

    @property
    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.environment.lower() in ("test", "testing", "ci")

    def get_model_path(self, version: Optional[str] = None) -> Path:
        """Get path to model file."""
        if version:
            return self.models_dir / f"model_{version}.pkl"
        return self.models_dir / "model_current.pkl"

    def get_feature_definitions_path(self) -> Path:
        """Get full path to feature definitions file."""
        return BASE_DIR / self.feature_definitions_path

    @property
    def model_version(self) -> str:
        """Get model version for backward compatibility."""
        return self.ml_model_version

    def get_cors_origins_list(self) -> List[str]:
        """Get CORS origins as a list."""
        if isinstance(self.cors_origins, str):
            return self.assemble_cors_origins(self.cors_origins)
        return self.cors_origins


# Create global settings instance
settings = Settings()

# Create necessary directories
settings.models_dir.mkdir(parents=True, exist_ok=True)
settings.synthetic_data_dir.mkdir(parents=True, exist_ok=True)
(BASE_DIR / "data").mkdir(exist_ok=True)
(BASE_DIR / "logs").mkdir(exist_ok=True)
