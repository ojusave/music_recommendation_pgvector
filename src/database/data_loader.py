"""Data Loader - Modified to work WITHOUT pandas (saves 50-80MB RAM)."""

import os, tempfile, csv, logging
from typing import List, Dict, Optional
from ..config import Config

logger = logging.getLogger(__name__)

class DataLoader:
    """Loads music data from Kaggle WITHOUT pandas to save memory."""
    
    def __init__(self, max_songs: int = None):
        # Use optimized default from config if not specified
        self.max_songs = max_songs or Config.MAX_KAGGLE_SONGS
    
    def load_kaggle_dataset(self) -> List[Dict]:
        """Load dataset without pandas to save memory."""
        if not Config.KAGGLE_USERNAME or not Config.KAGGLE_KEY:
            logger.warning("Kaggle credentials not configured - using sample data")
            return []
        
        try:
            import kaggle
            os.environ.update({'KAGGLE_USERNAME': Config.KAGGLE_USERNAME, 'KAGGLE_KEY': Config.KAGGLE_KEY})
            logger.info("Loading music dataset from Kaggle (NO pandas)...")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                return self._download_and_process_dataset(temp_dir)
        except ImportError:
            logger.error("Kaggle package not installed")
            return []
        except Exception as e:
            logger.error(f"Failed to load Kaggle dataset: {e}")
            return []
    
    def _download_and_process_dataset(self, temp_dir: str) -> List[Dict]:
        """Download and process without pandas."""
        try:
            import kaggle
            kaggle.api.dataset_download_files("usasha/million-music-playlists", path=temp_dir, unzip=True)
            
            track_file = next((os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f == 'track_meta.tsv'), None)
            return self._process_metadata_file_no_pandas(track_file) if track_file else []
            
        except Exception as e:
            logger.error(f"Dataset download failed: {e}")
            return []
    
    def _process_metadata_file_no_pandas(self, track_meta_file: str) -> List[Dict]:
        """Process metadata file using CSV module instead of pandas."""
        logger.info(f"Processing tracks without pandas (memory efficient)")
        
        songs = []
        processed_tracks = set()
        song_id = 1
        
        try:
            with open(track_meta_file, 'r', encoding='utf-8', errors='ignore') as f:
                # Use csv.DictReader instead of pandas
                reader = csv.DictReader(f, delimiter='\t')
                
                for row_num, row in enumerate(reader):
                    if len(songs) >= self.max_songs:
                        break
                    
                    try:
                        song = self._process_single_track_no_pandas(row, song_id, processed_tracks)
                        if song:
                            songs.append(song)
                            song_id += 1
                    except Exception as e:
                        continue
                        
                    # Progress logging
                    if row_num % 1000 == 0:
                        logger.info(f"Processed {row_num} rows, got {len(songs)} songs")
                        
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            return []
        
        logger.info(f"Processed {len(songs)} unique songs without pandas")
        return songs
    
    def _process_single_track_no_pandas(self, row: dict, song_id_counter: int, processed_tracks: set) -> Optional[Dict]:
        """Process single track without pandas."""
        # Get values with fallbacks
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
            'description': self._create_enhanced_description_no_pandas(track_name, artist_name, row)
        }
    
    def _create_enhanced_description_no_pandas(self, track_name: str, artist_name: str, row: dict) -> str:
        """Create enhanced description without pandas."""
        parts = [f"{track_name} by {artist_name}", f"artist: {artist_name}", f"song: {track_name}"]
        
        # Add metadata if available - check for None and empty strings
        for col_group, prefix in [(['album_name', 'album'], 'album'), (['genre', 'genres', 'style'], 'genre')]:
            for col in col_group:
                value = row.get(col)
                if value and str(value).strip() and str(value).strip().lower() not in ['nan', 'none', '']:
                    parts.append(f"{prefix}: {str(value).strip()}")
                    break
        
        # Add year/decade
        for col in ['year', 'release_year', 'date']:
            value = row.get(col)
            if value and str(value).strip() and str(value).strip().isdigit():
                year = str(value).strip()
                if len(year) == 4:
                    parts.append(f"{year[:3]}0s music")
                    break
        
        return " - ".join(parts)[:500]