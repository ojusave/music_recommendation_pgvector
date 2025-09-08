"""Database Schema Manager - PostgreSQL schema creation and management."""

import asyncpg, logging
from typing import Optional

logger = logging.getLogger(__name__)

class SchemaManager:
    """Manages PostgreSQL schema creation with pgvector extension and vector indexes."""
    
    def __init__(self, vector_dimensions: int = 768):
        self.vector_dimensions = vector_dimensions
    
    async def create_database_schema(self, connection_pool):
        """Create songs table with pgvector extension and indexes."""
        logger.info("Creating database schema...")
        
        async with connection_pool.acquire() as conn:
            # Enable pgvector extension and drop existing table
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            await conn.execute("DROP TABLE IF EXISTS songs CASCADE;")
            
            # Create songs table with vector column
            create_table_query = f"""
            CREATE TABLE songs (
                id SERIAL PRIMARY KEY,
                song_id TEXT UNIQUE NOT NULL,
                song_name TEXT NOT NULL,
                band TEXT NOT NULL,
                description TEXT NOT NULL,
                embedding VECTOR({self.vector_dimensions}),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            await conn.execute(create_table_query)
            await self._create_indexes(conn)
            
            logger.info(f"Database schema created with {self.vector_dimensions}-dimensional vectors")
    
    async def _create_indexes(self, conn):
        """Create vector and metadata indexes for optimal performance."""
        # IVFFlat index for vector similarity search
        await conn.execute("""
            CREATE INDEX songs_embedding_idx 
            ON songs USING ivfflat (embedding vector_cosine_ops) 
            WITH (lists = 100);
        """)
        
        # Traditional indexes for metadata searching
        await conn.execute("CREATE INDEX songs_song_name_idx ON songs (song_name);")
        await conn.execute("CREATE INDEX songs_band_idx ON songs (band);")
        
        logger.info("Database indexes created")
    
    async def verify_schema(self, connection_pool) -> bool:
        """Verify that the database schema exists and is correct."""
        try:
            async with connection_pool.acquire() as conn:
                # Check if songs table exists
                result = await conn.fetchval("""
                    SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'songs');
                """)
                
                if not result:
                    logger.warning("Songs table does not exist")
                    return False
                
                # Check vector dimension
                dimension_result = await conn.fetchval("""
                    SELECT atttypmod FROM pg_attribute 
                    WHERE attrelid = 'songs'::regclass AND attname = 'embedding';
                """)
                
                if dimension_result != self.vector_dimensions:
                    logger.warning(f"Vector dimension mismatch: expected {self.vector_dimensions}, got {dimension_result}")
                    return False
                
                logger.info("Database schema verification passed")
                return True
                
        except Exception as e:
            logger.error(f"Schema verification failed: {e}")
            return False
    
    async def get_table_stats(self, connection_pool) -> dict:
        """Get statistics about the songs table."""
        try:
            async with connection_pool.acquire() as conn:
                count = await conn.fetchval("SELECT COUNT(*) FROM songs")
                size_result = await conn.fetchrow("""
                    SELECT 
                        pg_size_pretty(pg_total_relation_size('songs')) as total_size,
                        pg_size_pretty(pg_relation_size('songs')) as table_size,
                        pg_size_pretty(pg_indexes_size('songs')) as indexes_size
                """)
                
                return {
                    'row_count': count,
                    'total_size': size_result['total_size'],
                    'table_size': size_result['table_size'], 
                    'indexes_size': size_result['indexes_size'],
                    'vector_dimensions': self.vector_dimensions
                }
                
        except Exception as e:
            logger.error(f"Failed to get table stats: {e}")
            return {'error': str(e)}
