"""Embedding Processor - Handles embedding generation and database insertion."""

import asyncpg, numpy as np, logging
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from ..config import Config

logger = logging.getLogger(__name__)

class EmbeddingProcessor:
    """Handles embedding generation, normalization, and batch database insertion."""
    
    def __init__(self, model: SentenceTransformer):
        self.model = model
    
    def generate_embeddings(self, descriptions: List[str]) -> np.ndarray:
        """Generate normalized embeddings for song descriptions."""
        logger.info(f"Generating embeddings for {len(descriptions)} descriptions...")
        
        embeddings = self.model.encode(descriptions)
        
        # Normalize embeddings to unit length for proper cosine similarity
        for i in range(len(embeddings)):
            embeddings[i] = embeddings[i] / np.linalg.norm(embeddings[i])
        
        logger.info("Embeddings generated and normalized")
        return embeddings
    
    async def insert_songs_with_embeddings(self, songs: List[Dict], embeddings: np.ndarray, connection_pool=None):
        """Insert songs with embeddings using connection pool or direct connection."""
        if connection_pool:
            async with connection_pool.acquire() as conn:
                await self._batch_insert(songs, embeddings, conn)
        else:
            conn = await asyncpg.connect(Config.DATABASE_URL)
            try:
                await self._batch_insert(songs, embeddings, conn)
            finally:
                await conn.close()
    
    async def _batch_insert(self, songs: List[Dict], embeddings: np.ndarray, conn):
        """Perform batch insertion of songs and embeddings."""
        insert_query = """
        INSERT INTO songs (song_id, song_name, band, description, embedding)
        VALUES ($1, $2, $3, $4, $5::vector)
        """
        
        batch_size = 100
        total_batches = (len(songs) - 1) // batch_size + 1
        
        for i in range(0, len(songs), batch_size):
            batch_songs = songs[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            
            for j, song in enumerate(batch_songs):
                embedding_str = '[' + ','.join(map(str, batch_embeddings[j].tolist())) + ']'
                await conn.execute(
                    insert_query,
                    song['song_id'], song['song_name'], song['band'],
                    song['description'], embedding_str
                )
            
            logger.info(f"Loaded batch {i//batch_size + 1}/{total_batches}")
        
        logger.info(f"Successfully inserted {len(songs)} songs with embeddings")
    
    async def insert_sample_songs(self, songs: List[Dict], connection_pool):
        """Insert sample songs with embeddings (optimized for smaller datasets)."""
        logger.info("Processing sample songs...")
        
        descriptions = [song['description'] for song in songs]
        embeddings = self.generate_embeddings(descriptions)
        
        async with connection_pool.acquire() as conn:
            insert_query = """
            INSERT INTO songs (song_id, song_name, band, description, embedding)
            VALUES ($1, $2, $3, $4, $5::vector)
            """
            
            for i, song in enumerate(songs):
                embedding_str = '[' + ','.join(map(str, embeddings[i].tolist())) + ']'
                await conn.execute(
                    insert_query, song['song_id'], song['song_name'], 
                    song['band'], song['description'], embedding_str
                )
        
        logger.info(f"Loaded {len(songs)} sample songs")
    
    def get_embedding_stats(self, embeddings: np.ndarray) -> Dict:
        """Get statistics about the generated embeddings."""
        return {
            'count': len(embeddings),
            'dimensions': embeddings.shape[1] if len(embeddings) > 0 else 0,
            'mean_norm': float(np.mean([np.linalg.norm(emb) for emb in embeddings])),
            'std_norm': float(np.std([np.linalg.norm(emb) for emb in embeddings])),
            'memory_usage_mb': float(embeddings.nbytes / (1024 * 1024))
        }
