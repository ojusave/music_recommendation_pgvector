"""
Database Setup Module for Music Recommendation App
=================================================

This module handles automatic database schema creation and sample data loading.
It demonstrates how to set up pgvector with PostgreSQL.
"""

import asyncpg
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict

from .config import Config

logger = logging.getLogger(__name__)

class DatabaseSetup:
    """
    Handles automatic database initialization and sample data loading.
    
    This class demonstrates:
    1. Creating PostgreSQL tables with pgvector extension
    2. Setting up vector indexes for optimal performance
    3. Loading sample data with embeddings for immediate functionality
    """
    
    def __init__(self, model: SentenceTransformer):
        self.model = model
    
    async def create_database_schema(self, connection_pool):
        """
        Create the songs table with proper schema for pgvector.
        
        This demonstrates:
        - Enabling pgvector extension
        - Creating tables with VECTOR columns
        - Setting up indexes for fast similarity search
        """
        logger.info("Creating database schema...")
        
        async with connection_pool.acquire() as conn:
            # Enable pgvector extension - required for vector operations
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Drop existing table if it exists (for clean setup)
            await conn.execute("DROP TABLE IF EXISTS songs CASCADE;")
            
            # Create songs table with vector column
            # Vector dimension (384) must match your embedding model
            create_table_query = """
            CREATE TABLE songs (
                id SERIAL PRIMARY KEY,
                song_id TEXT UNIQUE NOT NULL,
                song_name TEXT NOT NULL,
                band TEXT NOT NULL,
                description TEXT NOT NULL,
                embedding VECTOR(384),  -- 384 dimensions for paraphrase-MiniLM-L3-v2
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            await conn.execute(create_table_query)
            
            # Create indexes for optimal performance
            # IVFFlat index with cosine distance for vector similarity search
            await conn.execute("""
                CREATE INDEX songs_embedding_idx 
                ON songs USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = 100);
            """)
            
            # Traditional indexes for metadata searching
            await conn.execute("CREATE INDEX songs_song_name_idx ON songs (song_name);")
            await conn.execute("CREATE INDEX songs_band_idx ON songs (band);")
            
            logger.info("Database schema created")
    
    async def load_sample_data(self, connection_pool):
        """
        Load curated sample songs with enhanced descriptions.
        
        This demonstrates:
        1. Creating rich, semantic descriptions for better search
        2. Generating embeddings using sentence transformers
        3. Batch inserting vectors into PostgreSQL
        """
        logger.info("Loading sample music data...")
        
        # Curated sample songs with enhanced semantic descriptions
        # These descriptions include genre, mood, and context for better search
        sample_songs = [
            {
                'song_id': '1',
                'song_name': 'Bohemian Rhapsody',
                'band': 'Queen',
                'description': 'Bohemian Rhapsody by Queen - rock, epic, operatic'
            },
            {
                'song_id': '2', 
                'song_name': 'Hotel California',
                'band': 'Eagles',
                'description': 'Hotel California by Eagles - rock, classic, mysterious'
            },
            {
                'song_id': '3',
                'song_name': 'Imagine',
                'band': 'John Lennon',
                'description': 'Imagine by John Lennon - peaceful, hopeful, ballad'
            },
            {
                'song_id': '4',
                'song_name': 'Stairway to Heaven',
                'band': 'Led Zeppelin',
                'description': 'Stairway to Heaven by Led Zeppelin - rock, epic, spiritual'
            },
            {
                'song_id': '5',
                'song_name': 'Purple Rain',
                'band': 'Prince',
                'description': 'Purple Rain by Prince - ballad, emotional, rainy day music'
            },
            {
                'song_id': '6',
                'song_name': 'Sweet Child O Mine',
                'band': 'Guns N Roses',
                'description': 'Sweet Child O Mine by Guns N Roses - rock, upbeat, energetic'
            },
            {
                'song_id': '7',
                'song_name': 'Yesterday',
                'band': 'The Beatles',
                'description': 'Yesterday by The Beatles - sad, ballad, melancholic'
            },
            {
                'song_id': '8',
                'song_name': 'Dancing Queen',
                'band': 'ABBA',
                'description': 'Dancing Queen by ABBA - dance, party, upbeat, disco'
            },
            {
                'song_id': '9',
                'song_name': 'The Sound of Silence',
                'band': 'Simon and Garfunkel',
                'description': 'The Sound of Silence by Simon and Garfunkel - melancholic, contemplative, folk'
            },
            {
                'song_id': '10',
                'song_name': 'Smells Like Teen Spirit',
                'band': 'Nirvana',
                'description': 'Smells Like Teen Spirit by Nirvana - rock, grunge, energetic, workout music'
            }
        ]
        
        # Generate embeddings for all descriptions
        # This is the core of semantic search - converting text to vectors
        descriptions = [song['description'] for song in sample_songs]
        embeddings = self.model.encode(descriptions)
        
        # Insert songs and embeddings into database
        async with connection_pool.acquire() as conn:
            insert_query = """
            INSERT INTO songs (song_id, song_name, band, description, embedding)
            VALUES ($1, $2, $3, $4, $5::vector)
            """
            
            for i, song in enumerate(sample_songs):
                # Convert numpy array to PostgreSQL vector format
                embedding_str = '[' + ','.join(map(str, embeddings[i].tolist())) + ']'
                await conn.execute(
                    insert_query,
                    song['song_id'],
                    song['song_name'], 
                    song['band'],
                    song['description'],
                    embedding_str
                )
        
        logger.info(f"Loaded {len(sample_songs)} sample songs")
    
    async def setup_database(self, connection_pool):
        """
        Complete database setup process.
        
        This is the main entry point for automatic database initialization.
        """
        logger.info("Starting automatic database setup...")
        
        try:
            # Step 1: Create schema and indexes
            await self.create_database_schema(connection_pool)
            
            # Step 2: Load sample data with embeddings
            await self.load_sample_data(connection_pool)
            
            logger.info("Automatic database setup completed!")
            
        except Exception as e:
            logger.error(f"Auto-setup failed: {e}")
            raise
