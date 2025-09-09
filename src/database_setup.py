"""Database Setup - Main orchestrator for database initialization and data loading."""

import logging
from sentence_transformers import SentenceTransformer
from .database import SchemaManager, DataLoader, EmbeddingProcessor, get_sample_songs, get_legacy_sample_songs

logger = logging.getLogger(__name__)

class DatabaseSetup:
    """Main orchestrator coordinating SchemaManager, DataLoader, and EmbeddingProcessor."""
    
    def __init__(self, model: SentenceTransformer, max_songs: int = 10000, use_halfvec: bool = True):
        self.model = model
        self.max_songs = max_songs
        self.use_halfvec = use_halfvec
        
        # Initialize specialized components with optimization settings
        model_dimensions = model.get_sentence_embedding_dimension()
        self.schema_manager = SchemaManager(vector_dimensions=model_dimensions, use_halfvec=use_halfvec)
        self.data_loader = DataLoader(max_songs=max_songs)
        # Note: EmbeddingProcessor will get updated vector type after schema creation
        self.embedding_processor = EmbeddingProcessor(model=model, use_halfvec=use_halfvec)
        
        logger.info(f"Database setup initialized with {model_dimensions}D embeddings, max {max_songs} songs")
    
    async def setup_database(self, connection_pool):
        """Complete database setup: schema creation and data loading."""
        logger.info("Starting automatic database setup...")
        
        try:
            # Create schema and indexes (this will detect pgvector capabilities)
            await self.schema_manager.create_database_schema(connection_pool)
            
            # Update embedding processor with detected vector type
            self.embedding_processor.use_halfvec = self.schema_manager.use_halfvec
            self.embedding_processor.vector_type = self.schema_manager.vector_type
            
            # Load music data (Kaggle if available, sample data otherwise)
            await self.load_music_data(connection_pool)
            
            # Verify setup
            stats = await self.schema_manager.get_table_stats(connection_pool)
            logger.info(f"Database setup completed! Stats: {stats}")
            
        except Exception as e:
            logger.error(f"Auto-setup failed: {e}")
            raise
    
    async def load_music_data(self, connection_pool):
        """Load music data from Kaggle first, fallback to sample data."""
        kaggle_songs = self.data_loader.load_kaggle_dataset()
        
        if kaggle_songs:
            logger.info(f"Using Kaggle dataset with {len(kaggle_songs)} songs")
            # Use streaming insertion for large datasets to optimize memory
            if len(kaggle_songs) > 1000:
                logger.info("Using streaming insertion for large dataset")
                await self.embedding_processor.insert_songs_streaming(kaggle_songs, connection_pool)
            else:
                await self._process_and_insert_songs(kaggle_songs, connection_pool)
        else:
            logger.info("Kaggle dataset not available, using sample data")
            sample_songs = get_legacy_sample_songs()
            await self.embedding_processor.insert_sample_songs(sample_songs, connection_pool)
    
    async def _process_and_insert_songs(self, songs, connection_pool):
        """Process songs and insert them with embeddings (for smaller datasets)."""
        descriptions = [song['description'] for song in songs]
        embeddings = self.embedding_processor.generate_embeddings(descriptions)
        
        stats = self.embedding_processor.get_embedding_stats(embeddings)
        logger.info(f"Embedding stats: {stats}")
        
        await self.embedding_processor.insert_songs_with_embeddings(songs, embeddings, connection_pool)
    
    async def load_sample_data_only(self, connection_pool, use_enhanced: bool = True):
        """Load only sample data (useful for testing or when Kaggle unavailable)."""
        logger.info("Loading sample music data...")
        
        sample_songs = get_sample_songs() if use_enhanced else get_legacy_sample_songs()
        await self.embedding_processor.insert_sample_songs(sample_songs, connection_pool)
    
    async def verify_setup(self, connection_pool) -> bool:
        """Verify that the database setup is correct and functional."""
        return await self.schema_manager.verify_schema(connection_pool)
    
    async def get_setup_stats(self, connection_pool) -> dict:
        """Get comprehensive statistics about the database setup."""
        stats = await self.schema_manager.get_table_stats(connection_pool)
        stats.update({
            'model_name': self.model.__class__.__name__,
            'max_songs_configured': self.max_songs,
            'vector_dimensions': self.model.get_sentence_embedding_dimension()
        })
        return stats
