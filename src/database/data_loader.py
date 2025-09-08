"""Data Loader - Handles Kaggle dataset loading and processing."""

import os, tempfile, pandas as pd, logging
from typing import List, Dict, Optional
from ..config import Config

logger = logging.getLogger(__name__)

class DataLoader:
    """Loads music data from Kaggle with rich descriptions for semantic search."""
    
    def __init__(self, max_songs: int = 10000):
        self.max_songs = max_songs
    
    def load_kaggle_dataset(self) -> List[Dict]:
        """Load Million Music Playlists dataset from Kaggle with enhanced descriptions."""
        if not Config.KAGGLE_USERNAME or not Config.KAGGLE_KEY:
            logger.warning("Kaggle credentials not configured")
            return []
        
        try:
            import kaggle
            os.environ.update({'KAGGLE_USERNAME': Config.KAGGLE_USERNAME, 'KAGGLE_KEY': Config.KAGGLE_KEY})
            logger.info("Loading music dataset from Kaggle...")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                return self._download_and_process_dataset(temp_dir)
        except ImportError:
            logger.error("Kaggle package not installed")
            return []
        except Exception as e:
            logger.error(f"Failed to load Kaggle dataset: {e}")
            return []
    
    def _download_and_process_dataset(self, temp_dir: str) -> List[Dict]:
        """Download and process Kaggle dataset."""
        try:
            import kaggle
            kaggle.api.dataset_download_files("usasha/million-music-playlists", path=temp_dir, unzip=True)
            
            track_file = next((os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f == 'track_meta.tsv'), None)
            return self._process_metadata_file(track_file) if track_file else []
            
        except Exception as e:
            logger.error(f"Dataset download failed: {e}")
            return []
    
    
    def _process_metadata_file(self, track_meta_file: str) -> List[Dict]:
        """Process metadata file into song dictionaries with enhanced descriptions."""
        track_df = pd.read_csv(track_meta_file, sep='\t')
        logger.info(f"Processing {len(track_df)} tracks")
        
        songs, processed_tracks, song_id = [], set(), 1
        
        for _, row in track_df.iterrows():
            if len(songs) >= self.max_songs:
                break
                
            try:
                song = self._process_single_track(row, song_id, processed_tracks)
                if song:
                    songs.append(song)
                    song_id += 1
            except Exception as e:
                continue
        
        logger.info(f"Processed {len(songs)} unique songs")
        return songs
    
    def _process_single_track(self, row, song_id_counter: int, processed_tracks: set) -> Optional[Dict]:
        """Process single track into song dictionary."""
        track_name = str(row.get('song_name', '')).strip()[:200]
        artist_name = str(row.get('band', '')).strip()[:100]
        
        if not track_name or not artist_name:
            return None
        
        track_key = f"{track_name.lower()}_{artist_name.lower()}"
        if track_key in processed_tracks:
            return None
        processed_tracks.add(track_key)
        
        return {
            'song_id': str(row.get('song_id', song_id_counter)),
            'song_name': track_name,
            'band': artist_name,
            'description': self._create_enhanced_description(track_name, artist_name, row)
        }
    
    def _create_enhanced_description(self, track_name: str, artist_name: str, row) -> str:
        """Create enhanced description for semantic search."""
        parts = [f"{track_name} by {artist_name}", f"artist: {artist_name}", f"song: {track_name}"]
        
        # Add metadata if available
        for col, prefix in [(['album_name', 'album'], 'album'), (['genre', 'genres', 'style'], 'genre')]:
            for c in col:
                if c in row and pd.notna(row[c]) and str(row[c]).strip():
                    parts.append(f"{prefix}: {str(row[c]).strip()}")
                    break
        
        # Add year/decade
        for col in ['year', 'release_year', 'date']:
            if col in row and pd.notna(row[col]):
                year = str(row[col]).strip()
                if year.isdigit() and len(year) == 4:
                    parts.append(f"{year[:3]}0s music")
                    break
        
        return " - ".join(parts)[:500]
