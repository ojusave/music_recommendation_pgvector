"""
Configuration Module for Music Recommendation App
================================================

This module centralizes all configuration settings for the application.
Developers can modify these settings to customize their deployment.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Configuration class that loads settings from environment variables.
    
    This demonstrates best practices for configuration management in
    production applications.
    """
    
    # Flask Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # Database Configuration
    DATABASE_URL = os.getenv('DATABASE_URL')
    
    # Memory Optimization Settings for 512MB
    OPTIMIZE_FOR_MEMORY = os.getenv('OPTIMIZE_FOR_MEMORY', 'false').lower() == 'true'
    
    # Model Configuration - Using ultra-small model for 512MB RAM constraint
    SENTENCE_TRANSFORMER_MODEL = os.getenv('SENTENCE_TRANSFORMER_MODEL', 'paraphrase-MiniLM-L3-v2')
    
    # Kaggle API Configuration (optional)
    KAGGLE_USERNAME = os.getenv('KAGGLE_USERNAME')
    KAGGLE_KEY = os.getenv('KAGGLE_KEY')
    
    # Database Pool Configuration - Optimized for 512MB constraint
    DB_MIN_POOL_SIZE = int(os.getenv('DB_MIN_POOL_SIZE', '1'))
    DB_MAX_POOL_SIZE = int(os.getenv('DB_MAX_POOL_SIZE', '1'))  # Single connection for memory
    DB_COMMAND_TIMEOUT = int(os.getenv('DB_COMMAND_TIMEOUT', '90'))
    
    # Application Configuration - Lower limits for memory
    APP_NAME = os.getenv('APP_NAME', 'music_recommendations')
    DEFAULT_RECOMMENDATION_LIMIT = int(os.getenv('DEFAULT_RECOMMENDATION_LIMIT', '3'))
    MAX_RECOMMENDATION_LIMIT = int(os.getenv('MAX_RECOMMENDATION_LIMIT', '5'))
    
    # Server Configuration
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', '5000'))
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Logging Configuration
    LOG_LEVEL = getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper())
    
    # Deployment Environment Detection
    IS_RENDER = os.getenv('RENDER') is not None
    IS_PRODUCTION = os.getenv('FLASK_ENV', '').lower() == 'production'
    
    # Model cache directory (use system temp in production)
    MODEL_CACHE_DIR = os.getenv('MODEL_CACHE_DIR', '/tmp/sentence_transformers' if IS_RENDER else './models')
    
    @classmethod
    def setup_memory_optimizations(cls):
        """Set up environment for minimal memory usage."""
        if cls.OPTIMIZE_FOR_MEMORY:
            # Disable parallelism to save memory
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            os.environ['NUMEXPR_MAX_THREADS'] = '1'
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logging.info("Memory optimizations applied for 512MB constraint")
    
    @classmethod
    def validate(cls):
        """
        Validate that all required configuration is present.
        
        This is important for deployments to catch configuration
        issues early in the deployment process.
        """
        errors = []
        
        if not cls.DATABASE_URL:
            errors.append("DATABASE_URL environment variable is required!")
        
        # Set up memory optimizations early
        cls.setup_memory_optimizations()
        
        # Memory-specific warnings
        if cls.OPTIMIZE_FOR_MEMORY:
            logging.warning("Running in 512MB memory-optimized mode")
            logging.warning("Some features may be limited to conserve memory")
        
        # Check if running on Render
        if cls.IS_RENDER:
            logging.info("Running on Render platform with memory optimizations")
            # Ensure proper port binding
            if not os.getenv('PORT'):
                logging.warning("PORT environment variable not set, using default")
        
        # Kaggle credentials are optional - will use sample data if not provided
        if cls.KAGGLE_USERNAME and cls.KAGGLE_KEY:
            logging.info("Kaggle credentials found - will load real music dataset")
        else:
            logging.info("Kaggle credentials not found - will use sample dataset")
        
        # Create model cache directory if it doesn't exist
        try:
            os.makedirs(cls.MODEL_CACHE_DIR, exist_ok=True)
            logging.info(f"Model cache directory: {cls.MODEL_CACHE_DIR}")
        except Exception as e:
            logging.warning(f"Could not create model cache directory: {e}")
        
        if errors:
            for error in errors:
                logging.error(error)
            sys.exit(1)
        
        logging.info(f"Configuration loaded successfully")
        logging.info(f"Environment: {'Production' if cls.IS_PRODUCTION else 'Development'}")
        logging.info(f"Model: {cls.SENTENCE_TRANSFORMER_MODEL}")
        logging.info(f"Database pool: {cls.DB_MIN_POOL_SIZE}-{cls.DB_MAX_POOL_SIZE}")
        logging.info(f"Memory optimized: {cls.OPTIMIZE_FOR_MEMORY}")
        logging.info(f"Port: {cls.PORT}, Host: {cls.HOST}")

# Initialize logging with configured level
logging.basicConfig(
    level=Config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Apply memory optimizations on import
Config.setup_memory_optimizations()

# Validate configuration on import
try:
    Config.validate()
except Exception as e:
    logger.error(f"Configuration validation failed: {e}")
    if Config.IS_PRODUCTION:
        sys.exit(1)  # Fail fast in production
    else:
        logger.warning("Continuing in development mode despite config issues")