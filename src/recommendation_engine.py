"""Music Recommendation Engine - Core semantic search using pgvector."""

import asyncio, asyncpg, numpy as np, logging, urllib.parse, gc
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from .config import Config
from .database_setup import DatabaseSetup

# Handle psycopg2 import gracefully
try:
    import psycopg2
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    logging.warning("psycopg2 not available, sync methods will be disabled")

logger = logging.getLogger(__name__)

class MusicRecommendationEngine:
    """Core recommendation engine using pgvector for semantic similarity search."""
    
    def __init__(self, use_halfvec: bool = True):
        self.model = None
        self.connection_pool = None
        self.database_setup = None
        self.use_halfvec = use_halfvec
        self.vector_type = "halfvec" if use_halfvec else "vector"
        
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
        
        # Initialize database setup and connection pool with optimized settings
        self.database_setup = DatabaseSetup(self.model, use_halfvec=self.use_halfvec)
        logger.info("Connecting to PostgreSQL with pgvector...")
        self.connection_pool = await asyncpg.create_pool(
            Config.DATABASE_URL,
            min_size=Config.DB_MIN_POOL_SIZE,
            max_size=Config.DB_MAX_POOL_SIZE,
            command_timeout=Config.DB_COMMAND_TIMEOUT,
            server_settings={'application_name': Config.APP_NAME}
        )
        
        await self._verify_and_setup_database()
        
        # Update vector type based on what was actually detected/supported
        self.use_halfvec = self.database_setup.schema_manager.use_halfvec
        self.vector_type = self.database_setup.schema_manager.vector_type
        
        logger.info("Music Recommendation Engine initialized successfully!")
        logger.info(f"Using vector type: {self.vector_type} (halfvec: {self.use_halfvec})")
    
    async def _verify_and_setup_database(self):
        """Verify database setup and auto-initialize if needed."""
        async with self.connection_pool.acquire() as conn:
            try:
                count = await conn.fetchval("SELECT COUNT(*) FROM songs")
                logger.info(f"Database connected! Found {count:,} songs")
                if count == 0:
                    logger.info("No songs found - loading sample data...")
                    await self.database_setup.setup_database(self.connection_pool)
            except Exception as e:
                if "does not exist" in str(e):
                    logger.info("Setting up database for first time...")
                    await self.database_setup.setup_database(self.connection_pool)
                else:
                    logger.error(f"Database connection failed. Is your DATABASE_URL correct? Error: {e}")
                    raise
    
    async def get_recommendations(self, query: str, limit: int = 5) -> List[Dict]:
        """Get music recommendations based on natural language query using pgvector similarity search."""
        if not self.model or not self.connection_pool:
            raise RuntimeError("Recommendation engine not initialized. Check your DATABASE_URL and model loading.")
        
        logger.info(f"Processing query: '{query}'")
        
        # Convert query to embedding and normalize it (same as database embeddings)
        query_embedding = self.model.encode(query)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize to unit length
        
        # Convert to half-precision if using halfvec
        if self.use_halfvec:
            query_embedding = query_embedding.astype(np.float16)
        
        query_vector_str = '[' + ','.join(map(str, query_embedding.tolist())) + ']'
        
        # Use L2 distance on normalized vectors with highly discriminating similarity scoring
        # Target: L2=0 → 100%, L2=0.8 → ~80%, L2=1.0 → ~35%, L2=1.2 → ~10%, L2=1.4+ → 0%
        # Formula: GREATEST(0, (100 - 65 * L2²)) - more aggressive than before
        search_query = f"""
        SELECT song_id, song_name, band, description,
               (embedding <-> $1::{self.vector_type}) as l2_distance,
               ROUND(CAST(GREATEST(0, 100.0 - 65.0 * POWER(embedding <-> $1::{self.vector_type}, 2)) AS numeric), 1) as similarity_score
        FROM songs 
        ORDER BY embedding <-> $1::{self.vector_type} ASC
        LIMIT $2
        """
        
        async with self.connection_pool.acquire() as conn:
            try:
                results = await conn.fetch(search_query, query_vector_str, limit)
            except Exception as e:
                if "does not exist" in str(e):
                    raise RuntimeError("Database tables not found. Make sure pgvector extension is enabled: CREATE EXTENSION vector;")
                if "vector" in str(e).lower():
                    raise RuntimeError("pgvector extension not installed. Run: CREATE EXTENSION IF NOT EXISTS vector;")
                logger.error(f"Database query failed: {e}")
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
                'l2_distance': round(float(row['l2_distance']), 4)
            }
            rec.update(self._generate_music_links(row['song_name'], row['band']))
            recommendations.append(rec)
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        
        # Force garbage collection to free memory
        gc.collect()
        
        return recommendations

    def get_recommendations_sync(self, query: str, limit: int = 5) -> List[Dict]:
        """Synchronous version of get_recommendations for Flask compatibility."""
        if not PSYCOPG2_AVAILABLE:
            logger.error("psycopg2 not available for sync operations")
            raise RuntimeError("Synchronous database operations not available. Please install psycopg2-binary.")
        
        logger.info(f"Processing query: '{query}'")
        
        # Generate query embedding and normalize it (same as database embeddings)
        query_embedding = self.model.encode(query)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize to unit length
        
        # Test halfvec support directly in sync method (more reliable than detection)
        vector_type = "vector"  # Safe default
        use_halfvec = False
        
        # Test if halfvec is supported by trying a simple cast
        try:
            test_conn = psycopg2.connect(Config.DATABASE_URL)
            test_cursor = test_conn.cursor()
            test_cursor.execute("SELECT '[1,2,3]'::halfvec(3);")
            test_cursor.close()
            test_conn.close()
            # If we get here, halfvec is supported
            vector_type = "halfvec"
            use_halfvec = True
            logger.info("Sync method: halfvec support confirmed via direct test")
        except Exception as e:
            logger.info(f"Sync method: halfvec not supported, using vector ({e})")
            vector_type = "vector"
            use_halfvec = False
        
        # Convert to half-precision if using halfvec
        if use_halfvec:
            query_embedding = query_embedding.astype(np.float16)
            
        query_vector_str = '[' + ','.join(map(str, query_embedding.tolist())) + ']'
        
        # SQL query using L2 distance on normalized vectors (equivalent to cosine distance)
        # Use highly discriminating similarity scoring (same as async method)
        # Target: L2=0 → 100%, L2=0.8 → ~80%, L2=1.0 → ~35%, L2=1.2 → ~10%, L2=1.4+ → 0%
        # Formula: GREATEST(0, (100 - 65 * L2²)) - more aggressive than before
        search_query = f"""
            SELECT song_name, band, 
                   (embedding <-> %s::{vector_type}) as l2_distance,
                   ROUND(CAST(GREATEST(0, 100.0 - 65.0 * POWER(embedding <-> %s::{vector_type}, 2)) AS numeric), 1) as similarity_score
            FROM songs 
            ORDER BY embedding <-> %s::{vector_type}
            LIMIT %s
        """
        
        # Use synchronous psycopg2 connection
        try:
            conn = psycopg2.connect(Config.DATABASE_URL)
            cursor = conn.cursor()
            cursor.execute(search_query, (query_vector_str, query_vector_str, query_vector_str, limit))
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            
            # Format results
            recommendations = []
            for row in results:
                song_name, band, l2_distance, similarity_score = row
                recommendations.append({
                    'song_name': song_name,
                    'artist': band,
                    'band': band,
                    'similarity_score': float(similarity_score),
                    'l2_distance': round(float(l2_distance), 4),
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
        if not PSYCOPG2_AVAILABLE:
            logger.warning("psycopg2 not available for sync stats")
            return {'status': 'error', 'message': 'psycopg2 not available'}
        
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