import os
import logging
import sys
from typing import List, Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # App Settings
    APP_NAME: str = "Security Query Assistant API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Database Settings
    DATABASE_USER: str = ""
    DATABASE_PASSWORD: str = ""
    DATABASE_HOST: str = "localhost"
    DATABASE_PORT: int = 5432
    DATABASE_NAME: str = ""
    DATABASE_URL: str = "sqlite:///./app.db"
    
    # Security
    SECRET_KEY: str = ""
    ADMIN_TOKEN: str = ""
    
    # AI Settings
    OPENAI_API_KEY: str = ""
    MODEL_VERSION: str = "0.1.0"
    LLM_MODEL: str = "gpt-5-nano"
    LLM_TEMPERATURE: float = 0.0
    FINETUNED_MODEL_NAME: str = ""
    
    # Query Settings
    MAX_QUERY_LENGTH: int = 1000
    
    # Batch Settings
    BATCH_MAX_SIZE: int = 100
    BATCH_MAX_CONCURRENT: int = 5
    
    # LangSmith Settings
    LANGSMITH_TRACING: bool = False
    LANGSMITH_ENDPOINT: str = ""
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = ""

    # CORS
    CORS_ORIGINS: List[str] = ["*"]
    
    OPENROUTER_API_KEY: str = ""

    OPENROUTER_PRIMARY_MODEL: str = ""  
    OPENROUTER_SECONDARY_MODEL: str = ""        
    OPENROUTER_FALLBACK_MODEL: str = "" 

    # Keep OpenAI models as options but not primary
    OPENROUTER_AVAILABLE_MODELS: List[str] = ["*"]

    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: Optional[str] = None
    LOG_MAX_SIZE: int = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT: int = 5
    LOG_TO_CONSOLE: bool = True
    LOG_TO_FILE: bool = False
    
    WHITELISTED_IPS: List[str] = []

    @field_validator('LOG_LEVEL')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'LOG_LEVEL must be one of {valid_levels}')
        return v.upper()
    
    @field_validator('DATABASE_PORT')
    def validate_database_port(cls, v):
        if isinstance(v, str) and v == "":
            return 5432
        return int(v) if v else 5432
    
    @property
    def log_level_int(self) -> int:
        """Get log level as integer."""
        return getattr(logging, self.LOG_LEVEL)
    
    class Config:
        env_file = ".env"
        case_sensitive = True


def setup_logging(settings: Settings) -> logging.Logger:
    """Configure application logging based on settings."""
    
    # Create logs directory if logging to file
    if settings.LOG_TO_FILE and settings.LOG_FILE:
        log_path = Path(settings.LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level_int)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(settings.LOG_FORMAT)
    
    # Console handler
    if settings.LOG_TO_CONSOLE:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(settings.log_level_int)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if settings.LOG_TO_FILE and settings.LOG_FILE:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            settings.LOG_FILE,
            maxBytes=settings.LOG_MAX_SIZE,
            backupCount=settings.LOG_BACKUP_COUNT
        )
        file_handler.setLevel(settings.log_level_int)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Create application logger
    app_logger = logging.getLogger(settings.APP_NAME)
    
    # Log configuration on startup
    app_logger.info(f"Logging configured - Level: {settings.LOG_LEVEL}")
    app_logger.info(f"Environment: {settings.ENVIRONMENT}")
    app_logger.info(f"Debug mode: {settings.DEBUG}")
    
    return app_logger


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Global settings instance
settings = get_settings()

# Setup logging
logger = setup_logging(settings)