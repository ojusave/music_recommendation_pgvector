"""
Kaggle Dataset Processing Pipeline
=================================

This script downloads the Million Music Playlists dataset from Kaggle,
samples 50K songs, generates embeddings, and inserts into Render PostgreSQL.

Process:
1. Download dataset from Kaggle using API
2. Sample 50K diverse songs from the full dataset
3. Generate vector embeddings using lightweight model
4. Insert into PostgreSQL with batch processing
"""

import os
import asyncio
import asyncpg
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import tempfile
import zipfile
import logging
from pathlib import Path
import kaggle
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DATABASE_URL = "postgresql://music_test_yqzs_user:kHY5R3cBHpvkVGFfNEZy3Fq3hj93ptsE@dpg-d2v73hvfte5s73bq2e30-a.oregon-postgres.render.com/music_test_yqzs"
KAGGLE_DATASET = "notshrirang/spotify-million-song-dataset"
SAMPLE_SIZE = 10000
BATCH_SIZE = 1000

# Kaggle API credentials
KAGGLE_CREDENTIALS = {
    "username": "johnmctavish",
    "key": "0e0a0a413457632518c7e035589fe07a"
}

class KaggleDataProcessor:
    def __init__(self):
        self.model = None
        self.connection = None
        
    async def setup(self):
        """Initialize model and database connection"""
        logger.info("Initializing processor...")
        
        # Setup Kaggle credentials
        self.setup_kaggle_credentials()
        
        # Load ultra-lightweight sentence transformer model for memory constraints
        logger.info("Loading sentence transformer model (paraphrase-MiniLM-L3-v2)...")
        self.model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
        logger.info(f"✅ Model loaded: {self.model.get_sentence_embedding_dimension()} dimensions")
        
        # Connect to database
        logger.info("Connecting to Render PostgreSQL...")
        self.connection = await asyncpg.connect(DATABASE_URL)
        logger.info("✅ Database connected")
    
    def setup_kaggle_credentials(self):
        """Setup Kaggle API credentials"""
        # Create .kaggle directory if it doesn't exist
        kaggle_dir = Path.home() / '.kaggle'
        kaggle_dir.mkdir(exist_ok=True)
        
        # Write credentials file
        credentials_file = kaggle_dir / 'kaggle.json'
        with open(credentials_file, 'w') as f:
            json.dump(KAGGLE_CREDENTIALS, f)
        
        # Set appropriate permissions
        credentials_file.chmod(0o600)
        logger.info("✅ Kaggle credentials configured")
    
    def download_dataset(self, temp_dir: str) -> str:
        """
        Download dataset from Kaggle
        
        Args:
            temp_dir: Temporary directory for download
            
        Returns:
            Path to extracted dataset file
        """
        logger.info(f"Downloading dataset: {KAGGLE_DATASET}")
        
        try:
            # Download dataset to temp directory
            kaggle.api.dataset_download_files(
                KAGGLE_DATASET,
                path=temp_dir,
                unzip=True
            )
            
            # Find the main data file
            data_files = list(Path(temp_dir).glob("*.csv"))
            if not data_files:
                raise FileNotFoundError("No CSV files found in downloaded dataset")
            
            # Use the largest CSV file (assuming it's the main dataset)
            main_file = max(data_files, key=lambda f: f.stat().st_size)
            logger.info(f"✅ Dataset downloaded: {main_file.name} ({main_file.stat().st_size / 1024 / 1024:.1f} MB)")
            
            return str(main_file)
            
        except Exception as e:
            logger.error(f"❌ Failed to download dataset: {e}")
            raise
    
    def sample_songs(self, file_path: str) -> pd.DataFrame:
        """
        Sample songs from the full dataset
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame with sampled songs
        """
        logger.info(f"Loading and sampling {SAMPLE_SIZE:,} songs from dataset...")
        
        try:
            # Read CSV with chunking to handle large file
            chunk_size = 10000
            sampled_songs = []
            total_processed = 0
            
            # Calculate sampling ratio
            # First, get total count
            total_songs = sum(1 for line in open(file_path)) - 1  # -1 for header
            sampling_ratio = min(SAMPLE_SIZE / total_songs, 1.0)
            
            logger.info(f"Total songs in dataset: {total_songs:,}")
            logger.info(f"Sampling ratio: {sampling_ratio:.4f}")
            
            # Process in chunks and collect more aggressively
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                # Clean column names
                chunk.columns = chunk.columns.str.strip()
                
                # Log columns info only once
                if total_processed == 0:
                    possible_id_cols = [col for col in chunk.columns if 'id' in col.lower()]
                    possible_name_cols = [col for col in chunk.columns if 'name' in col.lower() or 'song' in col.lower()]
                    possible_band_cols = [col for col in chunk.columns if 'band' in col.lower() or 'artist' in col.lower()]
                    
                    logger.info(f"Available columns: {list(chunk.columns)}")
                    logger.info(f"Possible ID columns: {possible_id_cols}")
                    logger.info(f"Possible name columns: {possible_name_cols}")
                    logger.info(f"Possible band columns: {possible_band_cols}")
                
                # Clean chunk first to see how many valid rows we have
                chunk_cleaned = self.clean_dataframe_chunk(chunk)
                
                # Sample from cleaned chunk
                if len(chunk_cleaned) > 0 and len(sampled_songs) < SAMPLE_SIZE:
                    # Take more rows to compensate for cleaning losses
                    sample_size_chunk = min(len(chunk_cleaned), SAMPLE_SIZE - len(sampled_songs))
                    if sample_size_chunk > 0:
                        if len(chunk_cleaned) > sample_size_chunk:
                            chunk_sample = chunk_cleaned.sample(n=sample_size_chunk, random_state=42)
                        else:
                            chunk_sample = chunk_cleaned
                        sampled_songs.append(chunk_sample)
                
                total_processed += len(chunk)
                current_sampled = sum(len(df) for df in sampled_songs) if sampled_songs else 0
                
                if current_sampled >= SAMPLE_SIZE:
                    logger.info(f"Reached target of {SAMPLE_SIZE:,} songs")
                    break
                
                if total_processed % 50000 == 0:
                    logger.info(f"Processed {total_processed:,} songs, collected {current_sampled:,} valid songs")
            
            # Combine all samples
            if sampled_songs:
                df = pd.concat(sampled_songs, ignore_index=True)
                df = df.head(SAMPLE_SIZE)  # Ensure exact count
            else:
                raise ValueError("No songs were sampled")
            
            # Final cleaning
            df = self.final_clean_dataframe(df)
            
            logger.info(f"✅ Sampled {len(df):,} songs")
            logger.info(f"Columns: {list(df.columns)}")
            logger.info(f"Sample data:\n{df.head()}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to sample songs: {e}")
            raise
    
    def clean_dataframe_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Clean a single chunk of data"""
        # Basic cleaning - remove rows with missing artist or song
        chunk = chunk.dropna(subset=['artist', 'song'])
        
        # Remove empty strings
        chunk = chunk[
            (chunk['artist'].astype(str).str.strip() != '') & 
            (chunk['song'].astype(str).str.strip() != '') &
            (chunk['artist'].astype(str).str.strip() != 'nan') & 
            (chunk['song'].astype(str).str.strip() != 'nan')
        ]
        
        # Apply column mapping
        chunk = chunk.rename(columns={'artist': 'band', 'song': 'song_name'})
        chunk['song_id'] = chunk.index.astype(str)
        
        # Enhanced description with semantic information
        chunk['description'] = chunk.apply(self.create_enhanced_description, axis=1)
        
        return chunk[['song_id', 'song_name', 'band', 'description']]
    
    def create_enhanced_description(self, row) -> str:
        """Create enhanced description with inferred semantic information"""
        song_name = str(row['song_name']).lower()
        band = str(row['band']).lower()
        
        # Initialize description with basic info
        description = f"{row['song_name']} by {row['band']}"
        
        # Infer genre from artist (basic heuristics)
        genre_tags = []
        if any(artist in band for artist in ['twitty', 'cash', 'nelson', 'williams', 'jones']):
            genre_tags.append('country')
        elif any(artist in band for artist in ['coldplay', 'radiohead', 'u2', 'oasis']):
            genre_tags.append('rock')
        elif any(artist in band for artist in ['sinatra', 'garland', 'bennett']):
            genre_tags.append('jazz')
        elif any(artist in band for artist in ['beyonce', 'swift', 'grande']):
            genre_tags.append('pop')
        
        # Infer mood from song title
        mood_tags = []
        if any(word in song_name for word in ['sad', 'cry', 'tears', 'lonely', 'blue', 'hurt', 'pain']):
            mood_tags.append('sad')
        elif any(word in song_name for word in ['happy', 'joy', 'dance', 'party', 'celebrate']):
            mood_tags.append('upbeat')
        elif any(word in song_name for word in ['love', 'heart', 'romantic', 'kiss']):
            mood_tags.append('romantic')
        elif any(word in song_name for word in ['rain', 'storm', 'cloudy', 'grey']):
            mood_tags.append('melancholic')
        
        # Infer song type
        type_tags = []
        if any(word in song_name for word in ['ballad', 'slow', 'tender']):
            type_tags.append('ballad')
        elif any(word in song_name for word in ['dance', 'beat', 'rhythm']):
            type_tags.append('dance')
        elif any(word in song_name for word in ['rock', 'roll']):
            type_tags.append('rock')
        
        # Infer context
        context_tags = []
        if any(word in song_name for word in ['rain', 'rainy', 'storm']):
            context_tags.append('rainy day music')
        elif any(word in song_name for word in ['work', 'job', 'morning']):
            context_tags.append('work music')
        elif any(word in song_name for word in ['night', 'evening', 'midnight']):
            context_tags.append('nighttime music')
        
        # Combine all tags
        all_tags = genre_tags + mood_tags + type_tags + context_tags
        
        if all_tags:
            description += f" - {', '.join(all_tags)}"
        
        return description
    
    def final_clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final cleaning and deduplication"""
        logger.info("Final cleaning...")
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['song_name', 'band'])
        
        # Reset song_id to be sequential
        df = df.reset_index(drop=True)
        df['song_id'] = df.index.astype(str)
        
        logger.info(f"✅ Final cleaned dataframe: {len(df):,} unique songs")
        return df
    
    def generate_embeddings(self, descriptions: List[str]) -> np.ndarray:
        """
        Generate vector embeddings for song descriptions
        
        Args:
            descriptions: List of song descriptions
            
        Returns:
            Array of embeddings
        """
        logger.info(f"Generating embeddings for {len(descriptions):,} songs...")
        
        try:
            # Generate embeddings in batches to manage memory
            all_embeddings = []
            
            for i in range(0, len(descriptions), BATCH_SIZE):
                batch = descriptions[i:i + BATCH_SIZE]
                batch_embeddings = self.model.encode(batch, show_progress_bar=False)
                all_embeddings.append(batch_embeddings)
                
                if (i + BATCH_SIZE) % 5000 == 0:
                    logger.info(f"Generated embeddings for {i + BATCH_SIZE:,}/{len(descriptions):,} songs")
            
            # Combine all embeddings
            embeddings = np.vstack(all_embeddings)
            
            logger.info(f"✅ Generated {len(embeddings):,} embeddings of {embeddings.shape[1]} dimensions")
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Failed to generate embeddings: {e}")
            raise
    
    async def insert_songs(self, df: pd.DataFrame, embeddings: np.ndarray):
        """
        Insert songs and embeddings into PostgreSQL
        
        Args:
            df: DataFrame with song data
            embeddings: Array of vector embeddings
        """
        logger.info(f"Inserting {len(df):,} songs into database...")
        
        try:
            # Prepare data for batch insert
            records = []
            for i, (_, row) in enumerate(df.iterrows()):
                # Convert embedding to string format for pgvector
                embedding_str = '[' + ','.join(map(str, embeddings[i].tolist())) + ']'
                record = (
                    row['song_id'],
                    row['song_name'],
                    row['band'],
                    row['description'],
                    embedding_str
                )
                records.append(record)
            
            # Insert in batches
            insert_query = """
            INSERT INTO songs (song_id, song_name, band, description, embedding)
            VALUES ($1, $2, $3, $4, $5::vector)
            ON CONFLICT (song_id) DO NOTHING
            """
            
            batch_count = 0
            for i in range(0, len(records), BATCH_SIZE):
                batch = records[i:i + BATCH_SIZE]
                await self.connection.executemany(insert_query, batch)
                batch_count += 1
                
                if batch_count % 10 == 0:
                    logger.info(f"Inserted batch {batch_count} ({i + len(batch):,}/{len(records):,} songs)")
            
            # Verify insertion
            total_count = await self.connection.fetchval("SELECT COUNT(*) FROM songs")
            logger.info(f"✅ Database now contains {total_count:,} songs")
            
        except Exception as e:
            logger.error(f"❌ Failed to insert songs: {e}")
            raise
    
    async def process_dataset(self):
        """Main processing pipeline"""
        logger.info("🎵 Starting Kaggle dataset processing pipeline...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Step 1: Download dataset
                dataset_file = self.download_dataset(temp_dir)
                
                # Step 2: Sample songs
                df = self.sample_songs(dataset_file)
                
                # Step 3: Generate embeddings
                embeddings = self.generate_embeddings(df['description'].tolist())
                
                # Step 4: Insert into database
                await self.insert_songs(df, embeddings)
                
                logger.info("🚀 Processing pipeline completed successfully!")
                
                # Show some stats
                await self.show_stats()
                
            except Exception as e:
                logger.error(f"❌ Pipeline failed: {e}")
                raise
    
    async def show_stats(self):
        """Show database statistics"""
        logger.info("📊 Database Statistics:")
        
        total_songs = await self.connection.fetchval("SELECT COUNT(*) FROM songs")
        logger.info(f"   Total songs: {total_songs:,}")
        
        # Sample some songs
        sample_songs = await self.connection.fetch("""
            SELECT song_name, band, description 
            FROM songs 
            ORDER BY RANDOM() 
            LIMIT 5
        """)
        
        logger.info("   Sample songs:")
        for song in sample_songs:
            logger.info(f"     • {song['song_name']} by {song['band']}")
    
    async def cleanup(self):
        """Clean up resources"""
        if self.connection:
            await self.connection.close()
            logger.info("✅ Database connection closed")

async def main():
    """Main function"""
    processor = KaggleDataProcessor()
    
    try:
        await processor.setup()
        await processor.process_dataset()
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        raise
    finally:
        await processor.cleanup()

if __name__ == "__main__":
    print("🎵 Music Recommendation App - Kaggle Data Processing")
    print("=" * 60)
    print(f"📊 Target: {SAMPLE_SIZE:,} songs")
    print(f"🤖 Model: paraphrase-MiniLM-L3-v2 (384 dimensions)")
    print(f"🗄️  Database: Render PostgreSQL")
    print("=" * 60)
    
    asyncio.run(main())