"""
Music Recommendation Web App
============================

A Flask web application that demonstrates how to use pgvector with Render's PostgreSQL
for semantic music search and recommendations.

This app allows users to:
1. Enter natural language queries for music recommendations
2. Get 2-5 relevant song suggestions with similarity scores
3. View results with optional YouTube/Spotify links

Built for deployment on Render with PostgreSQL + pgvector extension.

Author: Demo for Render customers
"""

import os
import sys
import asyncio
import asyncpg
import numpy as np
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict, Optional
import urllib.parse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
log_level = getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper())
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

# Configuration - use environment variables
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    logger.error("DATABASE_URL environment variable is required!")
    sys.exit(1)

class MusicRecommendationEngine:
    """
    Core recommendation engine using pgvector for semantic similarity search.
    
    This class demonstrates best practices for:
    - Loading and using sentence transformer models
    - Connecting to PostgreSQL with pgvector
    - Performing semantic similarity searches
    - Handling vector normalization issues
    """
    
    def __init__(self):
        self.model = None
        self.connection_pool = None
        
    async def initialize(self):
        """
        Initialize the recommendation engine.
        
        This method:
        1. Loads the same sentence transformer model used for data processing
        2. Creates a connection pool to PostgreSQL for efficient database access
        3. Verifies that the database contains song data
        """
        logger.info("Initializing Music Recommendation Engine...")
        
        # Load the sentence transformer model
        # Using the same model as data processing ensures consistent embeddings
        model_name = 'paraphrase-MiniLM-L3-v2'  # Ultra-lightweight model for 512MB limit
        logger.info(f"Loading sentence transformer model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        logger.info(f"Model loaded: {self.model.get_sentence_embedding_dimension()} dimensions")
        
        # Create connection pool for efficient database access
        logger.info("Connecting to PostgreSQL with pgvector...")
        min_pool_size = int(os.getenv('DB_MIN_POOL_SIZE', '1'))
        max_pool_size = int(os.getenv('DB_MAX_POOL_SIZE', '5'))
        command_timeout = int(os.getenv('DB_COMMAND_TIMEOUT', '30'))
        app_name = os.getenv('APP_NAME', 'music_recommendations')
        
        self.connection_pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=min_pool_size,
            max_size=max_pool_size,
            command_timeout=command_timeout,
            server_settings={
                'application_name': app_name,
            }
        )
        
        # Verify database setup
        async with self.connection_pool.acquire() as conn:
            try:
                count = await conn.fetchval("SELECT COUNT(*) FROM songs")
                logger.info(f"Database contains {count:,} songs")
                
                if count == 0:
                    logger.warning("No songs found in database. Run process_data.py first!")
            except Exception as e:
                logger.warning(f"Songs table not found or database not initialized: {e}")
                logger.warning("Database setup may be needed. Run database.py and process_data.py first!")
        
        logger.info("Music Recommendation Engine initialized successfully!")
    
    async def get_recommendations(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Get music recommendations based on natural language query.
        
        This method demonstrates:
        1. Converting text to embeddings using sentence transformers
        2. Performing similarity search with pgvector's cosine distance
        3. Handling vector normalization for consistent results
        4. Converting raw distances to meaningful similarity percentages
        
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
        # This query demonstrates proper pgvector usage:
        # 1. Uses cosine distance (<->) for similarity measurement
        # 2. Converts distance to intuitive similarity percentage using exponential decay
        # 3. Shows actual match quality with much better score distribution
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
        conn = await asyncpg.connect(DATABASE_URL)
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
        Generate YouTube and Spotify search links for songs (bonus feature).
        
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
        """Get database statistics for the admin/status page."""
        if not self.connection_pool:
            return {'error': 'Not connected to database'}
        
        try:
            conn = await asyncpg.connect(DATABASE_URL)
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

# Initialize the recommendation engine
recommendation_engine = MusicRecommendationEngine()

@app.route('/')
def index():
    """
    Main page - serves the music recommendation interface.
    
    This demonstrates a simple but effective UI for semantic search applications.
    """
    return render_template('index.html')

@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    """
    API endpoint for getting music recommendations.
    
    This endpoint demonstrates:
    1. Input validation for user queries
    2. Async database operations with proper error handling
    3. JSON API responses with structured data
    4. Performance logging for monitoring
    
    Expected JSON input:
    {
        "query": "upbeat rock music for working out",
        "limit": 5
    }
    
    Returns JSON response:
    {
        "success": true,
        "recommendations": [...],
        "query": "original query",
        "count": 5
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing query parameter'
            }), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({
                'success': False,
                'error': 'Query cannot be empty'
            }), 400
        
        # Validate limit parameter
        default_limit = int(os.getenv('DEFAULT_RECOMMENDATION_LIMIT', '5'))
        max_limit = int(os.getenv('MAX_RECOMMENDATION_LIMIT', '10'))
        limit = data.get('limit', default_limit)
        if not isinstance(limit, int) or limit < 1 or limit > max_limit:
            limit = default_limit
        
        # Get recommendations
        recommendations = asyncio.run(recommendation_engine.get_recommendations(query, limit))
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'query': query,
            'count': len(recommendations)
        })
        
    except Exception as e:
        logger.error(f"Recommendation API error: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@app.route('/api/status')
def api_status():
    """
    Status endpoint for monitoring database health and statistics.
    
    Useful for:
    - Monitoring database connectivity
    - Checking data availability
    - Debugging deployment issues
    """
    try:
        stats = asyncio.run(recommendation_engine.get_database_stats())
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Status API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Simple health check endpoint for Render deployment monitoring."""
    try:
        # Check if database is initialized
        stats = asyncio.run(recommendation_engine.get_database_stats())
        if 'error' in stats:
            return jsonify({
                'status': 'healthy', 
                'service': 'music-recommendations',
                'database': 'not_initialized',
                'message': 'App is running but database needs setup'
            })
        else:
            return jsonify({
                'status': 'healthy', 
                'service': 'music-recommendations',
                'database': 'ready',
                'songs': stats.get('total_songs', 0)
            })
    except Exception as e:
        return jsonify({
            'status': 'healthy', 
            'service': 'music-recommendations',
            'database': 'unknown',
            'message': 'App is running but database status unclear'
        })

def initialize_app():
    """Initialize the application components."""
    try:
        # Check if we're already in an event loop
        loop = asyncio.get_running_loop()
        # If we are, create a task instead
        loop.create_task(recommendation_engine.initialize())
    except RuntimeError:
        # No event loop running, safe to use asyncio.run()
        asyncio.run(recommendation_engine.initialize())

# Initialize the app when module is loaded
# But only if we're not being imported for testing
if __name__ != '__main__' and 'test' not in sys.modules:
    initialize_app()

if __name__ == '__main__':
    # For local development - initialize and run
    initialize_app()
    
    # Get configuration from environment variables
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '5000'))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    app.run(debug=debug, host=host, port=port)
