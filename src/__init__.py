"""
Music Recommendation App - Source Module
=======================================

This package contains the core components of the music recommendation system:

- config: Configuration management and environment variables
- recommendation_engine: Core semantic search and recommendation logic
- database_setup: Automatic database schema creation and sample data loading

This modular structure demonstrates best practices for organizing
production applications deployed on Render.
"""

from .config import Config
from .recommendation_engine import MusicRecommendationEngine
from .database_setup import DatabaseSetup

__all__ = ['Config', 'MusicRecommendationEngine', 'DatabaseSetup']
