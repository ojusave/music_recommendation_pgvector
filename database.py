"""
Database Setup for Music Recommendation App
==========================================

This script sets up the PostgreSQL database on Render with pgvector extension
and creates the required schema for storing songs with vector embeddings.
"""

import asyncpg
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Render PostgreSQL connection
DATABASE_URL = "postgresql://music_test_yqzs_user:kHY5R3cBHpvkVGFfNEZy3Fq3hj93ptsE@dpg-d2v73hvfte5s73bq2e30-a.oregon-postgres.render.com/music_test_yqzs"

async def setup_database():
    """
    Set up the database schema with pgvector extension.
    
    This function:
    1. Enables the pgvector extension
    2. Creates the songs table with vector column
    3. Creates indexes for efficient similarity search
    """
    logger.info("Connecting to Render PostgreSQL...")
    
    try:
        # Connect to the database
        connection = await asyncpg.connect(DATABASE_URL, ssl='require')
        logger.info("✅ Connected to database successfully")
        
        # Enable pgvector extension
        logger.info("Enabling pgvector extension...")
        await connection.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        logger.info("✅ pgvector extension enabled")
        
        # Check if pgvector is available
        result = await connection.fetchval(
            "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
        )
        if not result:
            raise RuntimeError("pgvector extension failed to install")
        
        # Drop existing table if it exists (for clean setup)
        logger.info("Dropping existing tables...")
        await connection.execute("DROP TABLE IF EXISTS songs CASCADE;")
        
        # Create songs table with vector column
        # Using 384-dimensional vectors from all-MiniLM-L6-v2 model (memory-efficient)
        logger.info("Creating songs table...")
        create_table_query = """
        CREATE TABLE songs (
            id SERIAL PRIMARY KEY,
            song_id TEXT UNIQUE NOT NULL,
            song_name TEXT NOT NULL,
            band TEXT NOT NULL,
            description TEXT NOT NULL,  -- Enhanced descriptions with genre/mood tags
            embedding VECTOR(384),      -- 384-dimensional vectors (fits in 512MB memory)
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        await connection.execute(create_table_query)
        logger.info("✅ Songs table created")
        
        # Create index for efficient vector similarity search
        # Using IVFFlat index with cosine distance for semantic similarity
        logger.info("Creating vector index...")
        index_query = """
        CREATE INDEX songs_embedding_idx 
        ON songs USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
        """
        await connection.execute(index_query)
        logger.info("✅ Vector index created")
        
        # Create additional indexes for metadata searching
        logger.info("Creating metadata indexes...")
        await connection.execute("CREATE INDEX songs_song_name_idx ON songs (song_name);")
        await connection.execute("CREATE INDEX songs_band_idx ON songs (band);")
        await connection.execute("CREATE INDEX songs_description_idx ON songs USING gin(to_tsvector('english', description));")
        logger.info("✅ Metadata indexes created")
        
        # Verify setup
        logger.info("Verifying database setup...")
        table_count = await connection.fetchval("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'songs'")
        index_count = await connection.fetchval("SELECT COUNT(*) FROM pg_indexes WHERE tablename = 'songs'")
        
        logger.info(f"✅ Database setup complete!")
        logger.info(f"   - Tables: {table_count}")
        logger.info(f"   - Indexes: {index_count}")
        logger.info(f"   - Ready for {50000:,} songs")
        
        await connection.close()
        
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        raise

async def test_connection():
    """Test basic database connectivity"""
    try:
        print("   Attempting to connect to database...")
        connection = await asyncpg.connect(
            DATABASE_URL, 
            ssl='require',
            command_timeout=30,
            server_settings={
                'application_name': 'music_app_test'
            }
        )
        print("   Connection established, testing query...")
        result = await connection.fetchval("SELECT version();")
        logger.info(f"✅ Database connection test successful")
        logger.info(f"   PostgreSQL version: {result}")
        await connection.close()
        return True
    except asyncio.TimeoutError:
        logger.error(f"❌ Database connection timeout - database may be suspended")
        logger.error(f"   Try accessing your Render dashboard to wake up the database")
        return False
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        logger.error(f"   Error type: {type(e).__name__}")
        
        # Check if it's a common issue
        error_str = str(e).lower()
        if "connection was closed" in error_str:
            logger.error(f"   This usually means the database is suspended or restarting")
            logger.error(f"   Go to your Render dashboard and check the database status")
        elif "connection refused" in error_str:
            logger.error(f"   Database may be down or connection string is incorrect")
        elif "timeout" in error_str:
            logger.error(f"   Database is taking too long to respond - may be suspended")
        
        return False

if __name__ == "__main__":
    print("🎵 Music Recommendation App - Database Setup")
    print("=" * 50)
    
    # Test connection first
    print("\n1. Testing database connection...")
    if asyncio.run(test_connection()):
        print("\n2. Setting up database schema...")
        asyncio.run(setup_database())
        print("\n🚀 Database setup complete! Ready for data processing.")
    else:
        print("\n❌ Cannot proceed without database connection.")