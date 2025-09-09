"""Music Recommendation Engine - Core semantic search using pgvector."""

import asyncio, asyncpg, numpy as np, logging, urllib.parse, gc, psycopg2
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from .config import Config
from .database_setup import DatabaseSetup

logger = logging.getLogger(__name__)

class MusicRecommendationEngine:
    """Core recommendation engine using pgvector for semantic similarity search."""
    
    def __init__(self):
        self.model = None
        self.connection_pool = None
        self.database_setup = None
        
    async def initialize(self):
        """Initialize the recommendation engine with model and database connection."""
        logger.info("Initializing Music Recommendation Engine...")
        
        # Load sentence transformer model (PyTorch only for memory efficiency)
        model_name = Config.SENTENCE_TRANSFORMER_MODEL
        logger.info(f"Loading model: {model_name}...")
        import os
        os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/tmp/sentence_transformers'
        # Force PyTorch backend only to save memory
        self.model = SentenceTransformer(model_name, device='cpu', cache_folder='/tmp/sentence_transformers')
        logger.info(f"Model loaded: {self.model.get_sentence_embedding_dimension()} dimensions")
        
        # Initialize database setup and connection pool
        self.database_setup = DatabaseSetup(self.model)
        logger.info("Connecting to PostgreSQL with pgvector...")
        self.connection_pool = await asyncpg.create_pool(
            Config.DATABASE_URL,
            min_size=Config.DB_MIN_POOL_SIZE,
            max_size=Config.DB_MAX_POOL_SIZE,
            command_timeout=Config.DB_COMMAND_TIMEOUT,
            server_settings={'application_name': Config.APP_NAME}
        )
        
        await self._verify_and_setup_database()
        logger.info("Music Recommendation Engine initialized successfully!")
    
    async def _verify_and_setup_database(self):
        """Verify database setup and auto-initialize if needed."""
        async with self.connection_pool.acquire() as conn:
            try:
                count = await conn.fetchval("SELECT COUNT(*) FROM songs")
                logger.info(f"Database contains {count:,} songs")
                if count == 0:
                    logger.warning("No songs found - auto-loading data...")
                    await self.database_setup.setup_database(self.connection_pool)
            except Exception as e:
                logger.warning(f"Songs table not found: {e}")
                logger.info("Auto-setting up database...")
                await self.database_setup.setup_database(self.connection_pool)
    
    async def get_recommendations(self, query: str, limit: int = 5) -> List[Dict]:
        """Get music recommendations based on natural language query using pgvector similarity search."""
        if not self.model or not self.connection_pool:
            raise RuntimeError("Recommendation engine not initialized")
        
        logger.info(f"Processing query: '{query}'")
        
        # Convert query to normalized embedding vector
        query_embedding = self.model.encode(query)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_vector_str = '[' + ','.join(map(str, query_embedding.tolist())) + ']'
        
        # Perform semantic similarity search using pgvector
        search_query = """
        SELECT song_id, song_name, band, description,
               (embedding <-> $1::vector) as raw_distance,
               ROUND(CAST(GREATEST(0, (1.0 - (embedding <-> $1::vector) / 2.0) * 100.0) AS numeric), 1) as similarity_score
        FROM songs 
        ORDER BY embedding <-> $1::vector ASC
        LIMIT $2
        """
        
        async with self.connection_pool.acquire() as conn:
            try:
                results = await conn.fetch(search_query, query_vector_str, limit)
            except Exception as e:
                if "does not exist" in str(e):
                    raise RuntimeError("Database not initialized")
                raise e
        
        # Convert results to API format with music service links
        recommendations = []
        for row in results:
            rec = {
                'song_id': row['song_id'],
                'song_name': row['song_name'],
                'artist': row['band'],
                'description': row['description'] or '',
                'similarity_score': round(float(row['similarity_score']), 1),
                'raw_distance': round(float(row['raw_distance']), 4)
            }
            rec.update(self._generate_music_links(row['song_name'], row['band']))
            recommendations.append(rec)
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        
        # Force garbage collection to free memory
        gc.collect()
        
        return recommendations

    def get_recommendations_sync(self, query: str, limit: int = 5) -> List[Dict]:
        """Synchronous version of get_recommendations for Flask compatibility."""
        logger.info(f"Processing query: '{query}'")
        
        # Generate query embedding
        query_vector = self.model.encode([query])
        query_vector_str = f"[{','.join(map(str, query_vector[0]))}]"
        
        # SQL query for semantic similarity search
        search_query = """
            SELECT song_name, band, 
                   ROUND(CAST(GREATEST(0, (1.0 - (embedding <-> %s::vector) / 2.0) * 100.0) AS numeric), 1) as similarity_score
            FROM songs 
            ORDER BY embedding <-> %s::vector 
            LIMIT %s
        """
        
        # Use synchronous psycopg2 connection
        try:
            conn = psycopg2.connect(Config.DATABASE_URL)
            cursor = conn.cursor()
            cursor.execute(search_query, (query_vector_str, query_vector_str, limit))
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            
            # Format results
            recommendations = []
            for row in results:
                song_name, band, similarity_score = row
                recommendations.append({
                    'song_name': song_name,
                    'artist': band,  # Frontend expects 'artist' field
                    'band': band,    # Keep 'band' for backward compatibility  
                    'similarity_score': float(similarity_score),
                    'youtube_url': f"https://www.youtube.com/results?search_query={urllib.parse.quote(f'{song_name} {band}')}",
                    'spotify_url': f"https://open.spotify.com/search/{urllib.parse.quote(f'{song_name} {band}')}"
                })
            
            logger.info(f"Generated {len(recommendations)} recommendations")
            gc.collect()
            return recommendations
            
        except Exception as e:
            logger.error(f"Sync recommendation error: {e}")
            raise
    
    def _generate_music_links(self, song_name: str, artist: str) -> Dict[str, str]:
        """Generate YouTube and Spotify search links for songs."""
        search_query = f"{song_name.strip()} {artist.strip()}"
        encoded_query = urllib.parse.quote_plus(search_query)
        return {
            'youtube_url': f"https://www.youtube.com/results?search_query={encoded_query}",
            'spotify_url': f"https://open.spotify.com/search/{encoded_query}"
        }
    
    async def get_database_stats(self) -> Dict:
        """Get database statistics for monitoring and health checks."""
        if not self.connection_pool:
            return {'error': 'Not connected to database'}
        
        try:
            async with self.connection_pool.acquire() as conn:
                total_songs = await conn.fetchval("SELECT COUNT(*) FROM songs")
                sample_artists = await conn.fetch("SELECT band FROM songs LIMIT 10")
                return {
                    'total_songs': total_songs,
                    'sample_artists': [row['band'] for row in sample_artists],
                    'status': 'healthy'
                }
        except Exception as e:
            logger.error(f"Database stats error: {e}")
            return {'error': str(e)}

    def get_database_stats_sync(self) -> Dict:
        """Synchronous version of get_database_stats for Flask compatibility."""
        try:
            conn = psycopg2.connect(Config.DATABASE_URL)
            cursor = conn.cursor()
            
            # Get total songs count
            cursor.execute("SELECT COUNT(*) FROM songs")
            total_songs = cursor.fetchone()[0]
            
            # Get sample artists
            cursor.execute("SELECT band FROM songs LIMIT 10")
            sample_artists = [row[0] for row in cursor.fetchall()]
            
            cursor.close()
            conn.close()
            
            return {
                'total_songs': total_songs,
                'sample_artists': sample_artists,
                'status': 'healthy'
            }
        except Exception as e:
            logger.error(f"Sync database stats error: {e}")
            return {'status': 'error', 'message': str(e)}
