"""
Configuration Module for Music Recommendation App
================================================

This module centralizes all configuration settings for the application.
Render customers can modify these settings to customize their deployment.
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
    production applications deployed on Render.
    """
    
    # Flask Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # Database Configuration
    DATABASE_URL = os.getenv('DATABASE_URL')
    
    # Model Configuration
    SENTENCE_TRANSFORMER_MODEL = os.getenv('SENTENCE_TRANSFORMER_MODEL', 'paraphrase-MiniLM-L3-v2')
    
    # Database Pool Configuration
    DB_MIN_POOL_SIZE = int(os.getenv('DB_MIN_POOL_SIZE', '1'))
    DB_MAX_POOL_SIZE = int(os.getenv('DB_MAX_POOL_SIZE', '5'))
    DB_COMMAND_TIMEOUT = int(os.getenv('DB_COMMAND_TIMEOUT', '30'))
    
    # Application Configuration
    APP_NAME = os.getenv('APP_NAME', 'music_recommendations')
    DEFAULT_RECOMMENDATION_LIMIT = int(os.getenv('DEFAULT_RECOMMENDATION_LIMIT', '5'))
    MAX_RECOMMENDATION_LIMIT = int(os.getenv('MAX_RECOMMENDATION_LIMIT', '10'))
    
    # Server Configuration
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', '5000'))
    DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Logging Configuration
    LOG_LEVEL = getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper())
    
    @classmethod
    def validate(cls):
        """
        Validate that all required configuration is present.
        
        This is important for Render deployments to catch configuration
        issues early in the deployment process.
        """
        if not cls.DATABASE_URL:
            logging.error("DATABASE_URL environment variable is required!")
            sys.exit(1)
        
        logging.info(f"Configuration loaded successfully")
        logging.info(f"Model: {cls.SENTENCE_TRANSFORMER_MODEL}")
        logging.info(f"Database pool: {cls.DB_MIN_POOL_SIZE}-{cls.DB_MAX_POOL_SIZE}")

# Initialize logging with configured level
logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Validate configuration on import
Config.validate()
