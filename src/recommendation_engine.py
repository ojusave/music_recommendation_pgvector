"""
Music Recommendation Engine Module
=================================

This module contains the core recommendation logic using pgvector for semantic search.
It demonstrates how to build a production-ready vector similarity search system.
"""

import asyncio
import asyncpg
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict, Optional
import urllib.parse

from .config import Config
from .database_setup import DatabaseSetup

logger = logging.getLogger(__name__)

class MusicRecommendationEngine:
    """
    Core recommendation engine using pgvector for semantic similarity search.
    
    This class demonstrates best practices for:
    - Loading and using sentence transformer models
    - Connecting to PostgreSQL with pgvector
    - Performing semantic similarity searches
    - Handling vector normalization and scoring
    """
    
    def __init__(self):
        self.model = None
        self.connection_pool = None
        self.database_setup = None
        
    async def initialize(self):
        """
        Initialize the recommendation engine.
        
        This method:
        1. Loads the sentence transformer model for embeddings
        2. Creates a connection pool to PostgreSQL for efficient database access
        3. Verifies database setup and auto-initializes if needed
        """
        logger.info("Initializing Music Recommendation Engine...")
        
        # Load the sentence transformer model
        # Using the same model as data processing ensures consistent embeddings
        model_name = Config.SENTENCE_TRANSFORMER_MODEL
        logger.info(f"Loading sentence transformer model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        logger.info(f"Model loaded: {self.model.get_sentence_embedding_dimension()} dimensions")
        
        # Initialize database setup helper
        self.database_setup = DatabaseSetup(self.model)
        
        # Create connection pool for efficient database access
        logger.info("Connecting to PostgreSQL with pgvector...")
        self.connection_pool = await asyncpg.create_pool(
            Config.DATABASE_URL,
            min_size=Config.DB_MIN_POOL_SIZE,
            max_size=Config.DB_MAX_POOL_SIZE,
            command_timeout=Config.DB_COMMAND_TIMEOUT,
            server_settings={
                'application_name': Config.APP_NAME,
            }
        )
        
        # Verify database setup and auto-initialize if needed
        await self._verify_and_setup_database()
        
        logger.info("Music Recommendation Engine initialized successfully!")
    
    async def _verify_and_setup_database(self):
        """
        Verify database setup and automatically initialize if needed.
        
        This demonstrates automatic database provisioning for zero-config deployment.
        """
        async with self.connection_pool.acquire() as conn:
            try:
                count = await conn.fetchval("SELECT COUNT(*) FROM songs")
                logger.info(f"Database contains {count:,} songs")
                
                if count == 0:
                    logger.warning("No songs found in database. Auto-loading data...")
                    await self.database_setup.setup_database(self.connection_pool)
            except Exception as e:
                logger.warning(f"Songs table not found: {e}")
                logger.info("Auto-setting up database and loading data...")
                await self.database_setup.setup_database(self.connection_pool)
    
    async def get_recommendations(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Get music recommendations based on natural language query.
        
        This method demonstrates:
        1. Converting text to embeddings using sentence transformers
        2. Performing similarity search with pgvector's cosine distance
        3. Converting raw distances to meaningful similarity percentages
        4. Returning structured results with metadata
        
        Args:
            query: Natural language description of desired music
            limit: Number of recommendations to return (2-5 recommended)
            
        Returns:
            List of song dictionaries with similarity scores and metadata
        """
        if not self.model or not self.connection_pool:
            raise RuntimeError("Recommendation engine not initialized")
        
        logger.info(f"Processing recommendation query: '{query}'")
        
        # Convert query to embedding vector
        # Important: Use the same model and normalization as your stored vectors
        query_embedding = self.model.encode(query)
        
        # Convert numpy array to PostgreSQL vector format
        # pgvector expects vectors as string arrays: '[1.0, 2.0, 3.0]'
        query_vector_str = '[' + ','.join(map(str, query_embedding.tolist())) + ']'
        
        # Perform semantic similarity search using pgvector
        # This query demonstrates proper pgvector usage with realistic scoring
        search_query = """
        SELECT 
            song_id,
            song_name,
            band,
            description,
            (embedding <-> $1::vector) as raw_distance,
            -- More realistic similarity calculation
            -- Converts cosine distance to percentage: Distance 0 = 100%, Distance 1 = 50%, Distance 2 = 0%
            ROUND(CAST(GREATEST(0, (2.0 - (embedding <-> $1::vector)) * 50.0) AS numeric), 1) as similarity_score
        FROM songs 
        ORDER BY embedding <-> $1::vector ASC
        LIMIT $2
        """
        
        # Use a fresh connection for each request to avoid pool issues
        conn = await asyncpg.connect(Config.DATABASE_URL)
        try:
            results = await conn.fetch(search_query, query_vector_str, limit)
        except Exception as e:
            await conn.close()
            if "does not exist" in str(e):
                raise RuntimeError("Database not initialized. Please set up the database first by running database.py and process_data.py")
            else:
                raise e
        finally:
            await conn.close()
        
        # Convert database results to API-friendly format
        recommendations = []
        for row in results:
            recommendation = {
                'song_id': row['song_id'],
                'song_name': row['song_name'],
                'artist': row['band'],
                'description': row['description'] or '',
                'similarity_score': round(float(row['similarity_score']), 1),
                'raw_distance': round(float(row['raw_distance']), 4)
            }
            
            # Add music service links (bonus feature)
            recommendation.update(self._generate_music_links(
                row['song_name'], 
                row['band']
            ))
            
            recommendations.append(recommendation)
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations
    
    def _generate_music_links(self, song_name: str, artist: str) -> Dict[str, str]:
        """
        Generate YouTube and Spotify search links for songs.
        
        Note: These are search links, not direct song links, as we don't have
        official music service IDs in our dataset. In a production app, you'd
        want to integrate with official APIs for accurate links.
        """
        # Clean up song and artist names for URL encoding
        clean_song = song_name.strip()
        clean_artist = artist.strip()
        
        # Create search query
        search_query = f"{clean_song} {clean_artist}"
        encoded_query = urllib.parse.quote_plus(search_query)
        
        return {
            'youtube_url': f"https://www.youtube.com/results?search_query={encoded_query}",
            'spotify_url': f"https://open.spotify.com/search/{encoded_query}"
        }
    
    async def get_database_stats(self) -> Dict:
        """
        Get database statistics for monitoring and health checks.
        
        This is useful for:
        - Monitoring database connectivity
        - Checking data availability
        - Debugging deployment issues
        """
        if not self.connection_pool:
            return {'error': 'Not connected to database'}
        
        try:
            conn = await asyncpg.connect(Config.DATABASE_URL)
            try:
                total_songs = await conn.fetchval("SELECT COUNT(*) FROM songs")
                
                # Get sample of artists to show data variety
                sample_artists = await conn.fetch("""
                    SELECT band 
                    FROM songs 
                    LIMIT 10
                """)
                
                return {
                    'total_songs': total_songs,
                    'sample_artists': [row['band'] for row in sample_artists],
                    'status': 'healthy'
                }
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Database stats error: {e}")
            return {'error': str(e)}
