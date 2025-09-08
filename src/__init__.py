"""
Music Recommendation App - Source Module
=======================================

This package contains the core components of the music recommendation system:

- config: Configuration management and environment variables
- recommendation_engine: Core semantic search and recommendation logic
- database_setup: Main orchestrator for database initialization
- database/: Database package containing modular components
  - schema_manager: Database schema creation and management
  - data_loader: External data source loading and processing
  - embedding_processor: Embedding generation and database operations
  - sample_data: Curated sample music data

This modular structure demonstrates best practices for organizing
production applications with separation of concerns.
"""

from .config import Config
from .recommendation_engine import MusicRecommendationEngine
from .database_setup import DatabaseSetup

# Database package components (used internally by DatabaseSetup)
from .database import SchemaManager, DataLoader, EmbeddingProcessor, get_sample_songs, get_legacy_sample_songs

__all__ = [
    'Config', 
    'MusicRecommendationEngine', 
    'DatabaseSetup',  # Now uses modular architecture internally
    'SchemaManager',
    'DataLoader', 
    'EmbeddingProcessor',
    'get_sample_songs',
    'get_legacy_sample_songs'
]
