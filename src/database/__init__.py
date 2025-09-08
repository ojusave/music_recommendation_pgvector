"""
Database Package
================

This package contains all database-related modules for the music recommendation system:

- schema_manager: Database schema creation and management
- data_loader: External data source loading and processing  
- embedding_processor: Embedding generation and database operations
- sample_data: Curated sample music data

This modular structure separates database concerns from the main application logic.
"""

from .schema_manager import SchemaManager
from .data_loader import DataLoader
from .embedding_processor import EmbeddingProcessor
from .sample_data import get_sample_songs, get_legacy_sample_songs

__all__ = [
    'SchemaManager',
    'DataLoader', 
    'EmbeddingProcessor',
    'get_sample_songs',
    'get_legacy_sample_songs'
]
