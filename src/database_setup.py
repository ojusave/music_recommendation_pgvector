"""
Database Setup Module for Music Recommendation App
=================================================

This module handles automatic database schema creation and sample data loading.
It demonstrates how to set up pgvector with PostgreSQL.
"""

import asyncpg
import numpy as np
import pandas as pd
import json
import os
import tempfile
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
    
    def load_kaggle_dataset(self) -> List[Dict]:
        """
        Load music dataset from Kaggle.
        
        This function downloads and processes a music dataset from Kaggle
        if credentials are available. Popular music datasets include:
        - spotify-dataset (contains track features, artists, genres)
        - music-dataset (contains song metadata)
        
        Returns:
            List of song dictionaries with metadata
        """
        if not Config.KAGGLE_USERNAME or not Config.KAGGLE_KEY:
            logger.warning("Kaggle credentials not configured - skipping Kaggle dataset")
            return []
        
        try:
            # Import kaggle here to avoid errors if not installed
            import kaggle
            
            # Configure Kaggle API credentials
            os.environ['KAGGLE_USERNAME'] = Config.KAGGLE_USERNAME
            os.environ['KAGGLE_KEY'] = Config.KAGGLE_KEY
            
            logger.info("Loading music dataset from Kaggle...")
            
            # Create temporary directory for dataset download
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download the Million Music Playlists dataset
                dataset_name = "usasha/million-music-playlists"
                
                try:
                    kaggle.api.dataset_download_files(
                        dataset_name, 
                        path=temp_dir, 
                        unzip=True
                    )
                    logger.info(f"Downloaded dataset: {dataset_name}")
                    
                    # Look for the track metadata file
                    track_meta_file = None
                    interaction_file = None
                    
                    for file in os.listdir(temp_dir):
                        if file == 'track_meta.tsv':
                            track_meta_file = os.path.join(temp_dir, file)
                        elif file == 'user_item_interaction.csv':
                            interaction_file = os.path.join(temp_dir, file)
                    
                    if not track_meta_file:
                        logger.error("track_meta.tsv not found in downloaded dataset")
                        return []
                    
                    logger.info("Processing Million Music Playlists dataset files...")
                    
                    # Load track metadata from TSV file
                    logger.info("Loading track metadata...")
                    track_df = pd.read_csv(track_meta_file, sep='\t')
                    logger.info(f"Loaded {len(track_df)} tracks from metadata file")
                    
                    # Load user interactions to get playlist context (optional)
                    interaction_df = None
                    if interaction_file and os.path.exists(interaction_file):
                        logger.info("Loading user interaction data...")
                        interaction_df = pd.read_csv(interaction_file)
                        logger.info(f"Loaded {len(interaction_df)} interactions")
                    
                    # Process tracks and create rich descriptions
                    songs = []
                    song_id_counter = 1
                    processed_tracks = set()  # To avoid duplicates
                    
                    for idx, row in track_df.iterrows():
                        # Limit processing to avoid memory issues
                        if len(songs) >= 2000:
                            break
                            
                        try:
                            # Extract track information from different possible column names
                            track_name = None
                            artist_name = None
                            
                            # Use the correct column names from the dataset
                            if 'song_name' in row and pd.notna(row['song_name']):
                                track_name = str(row['song_name']).strip()
                            
                            if 'band' in row and pd.notna(row['band']):
                                artist_name = str(row['band']).strip()
                            
                            # Skip if missing essential info
                            if not track_name or not artist_name or track_name == '' or artist_name == '':
                                continue
                            
                            # Create unique identifier to avoid duplicates
                            track_key = f"{track_name.lower()}_{artist_name.lower()}"
                            if track_key in processed_tracks:
                                continue
                            processed_tracks.add(track_key)
                            
                            # Clean up names and limit length
                            track_name = track_name[:200]
                            artist_name = artist_name[:100]
                            
                            # Create rich description for semantic search
                            description_parts = [f"{track_name} by {artist_name}"]
                            
                            # Add album information if available
                            for col in ['album_name', 'album']:
                                if col in row and pd.notna(row[col]) and str(row[col]).strip():
                                    description_parts.append(f"album: {str(row[col]).strip()}")
                                    break
                            
                            # Add genre information if available
                            for col in ['genre', 'genres', 'style']:
                                if col in row and pd.notna(row[col]) and str(row[col]).strip():
                                    description_parts.append(f"genre: {str(row[col]).strip()}")
                                    break
                            
                            # Add year information if available
                            for col in ['year', 'release_year', 'date']:
                                if col in row and pd.notna(row[col]):
                                    year = str(row[col]).strip()
                                    if year.isdigit() and len(year) == 4:
                                        decade = f"{year[:3]}0s"
                                        description_parts.append(f"{decade} music")
                                        break
                            
                            # Add duration context if available
                            for col in ['duration', 'duration_ms', 'length']:
                                if col in row and pd.notna(row[col]):
                                    try:
                                        duration = float(row[col])
                                        if 'ms' in col:
                                            duration = duration / 1000
                                        
                                        if duration > 300:  # 5+ minutes
                                            description_parts.append("long track")
                                        elif duration < 120:  # under 2 minutes
                                            description_parts.append("short track")
                                        break
                                    except:
                                        continue
                            
                            # Create final description
                            description = " - ".join(description_parts)[:500]
                            
                            # Use original song_id from dataset if available, otherwise generate one
                            original_song_id = str(row.get('song_id', song_id_counter))
                            
                            song = {
                                'song_id': original_song_id,
                                'song_name': track_name,
                                'band': artist_name,
                                'description': description
                            }
                            
                            songs.append(song)
                            song_id_counter += 1
                            
                        except Exception as track_e:
                            logger.warning(f"Skipping track {idx} due to error: {track_e}")
                            continue
                    
                    logger.info(f"Processed {len(songs)} unique songs from Million Music Playlists dataset")
                    return songs
                    
                except Exception as e:
                    logger.error(f"Failed to download dataset {dataset_name}: {e}")
                    logger.info("Will fallback to sample data instead")
                    return []
        
        except ImportError:
            logger.error("Kaggle package not installed. Install with: pip install kaggle")
            return []
        except Exception as e:
            logger.error(f"Failed to load Kaggle dataset: {e}")
            return []
    
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
        
        # Normalize embeddings to unit length for proper cosine similarity
        # This ensures consistent distance calculations with query embeddings
        for i in range(len(embeddings)):
            embeddings[i] = embeddings[i] / np.linalg.norm(embeddings[i])
        
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
    
    async def load_music_data(self, connection_pool):
        """
        Load music data from Kaggle if available, otherwise use sample data.
        
        This function tries to load real music data from Kaggle first,
        and falls back to sample data if Kaggle is not available.
        """
        # Try to load from Kaggle first
        kaggle_songs = self.load_kaggle_dataset()
        
        if kaggle_songs:
            logger.info(f"Using Kaggle dataset with {len(kaggle_songs)} songs")
            songs_to_load = kaggle_songs
        else:
            logger.info("Kaggle dataset not available, using sample data")
            # Load sample songs (reuse the sample data from load_sample_data)
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
            songs_to_load = sample_songs
        
        # Generate embeddings for all descriptions
        descriptions = [song['description'] for song in songs_to_load]
        logger.info(f"Generating embeddings for {len(descriptions)} songs...")
        embeddings = self.model.encode(descriptions)
        
        # Normalize embeddings to unit length for proper cosine similarity
        for i in range(len(embeddings)):
            embeddings[i] = embeddings[i] / np.linalg.norm(embeddings[i])
        
        # Insert songs and embeddings into database
        conn = await asyncpg.connect(Config.DATABASE_URL)
        try:
            insert_query = """
            INSERT INTO songs (song_id, song_name, band, description, embedding)
            VALUES ($1, $2, $3, $4, $5::vector)
            """
            
            batch_size = 100  # Process in batches for large datasets
            for i in range(0, len(songs_to_load), batch_size):
                batch_songs = songs_to_load[i:i+batch_size]
                batch_embeddings = embeddings[i:i+batch_size]
                
                for j, song in enumerate(batch_songs):
                    # Convert numpy array to PostgreSQL vector format
                    embedding_str = '[' + ','.join(map(str, batch_embeddings[j].tolist())) + ']'
                    await conn.execute(
                        insert_query,
                        song['song_id'],
                        song['song_name'], 
                        song['band'],
                        song['description'],
                        embedding_str
                    )
                
                logger.info(f"Loaded batch {i//batch_size + 1}/{(len(songs_to_load)-1)//batch_size + 1}")
        finally:
            await conn.close()
        
        logger.info(f"Successfully loaded {len(songs_to_load)} songs with embeddings")
    
    async def setup_database(self, connection_pool):
        """
        Complete database setup process.
        
        This is the main entry point for automatic database initialization.
        """
        logger.info("Starting automatic database setup...")
        
        try:
            # Step 1: Create schema and indexes
            await self.create_database_schema(connection_pool)
            
            # Step 2: Load music data (Kaggle if available, sample data otherwise)
            await self.load_music_data(connection_pool)
            
            logger.info("Automatic database setup completed!")
            
        except Exception as e:
            logger.error(f"Auto-setup failed: {e}")
            raise
