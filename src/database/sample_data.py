"""
Sample Data Definitions
=======================

Contains curated sample music data for the recommendation system.

This file demonstrates how to structure song data for semantic search.
Each song has a 'description' field that the AI uses to understand the song's meaning.

Tips for good descriptions:
- Include genre, mood, tempo
- Add descriptive words that users might search for
- Think about how someone would describe the song to a friend
"""

from typing import List, Dict

def get_sample_songs() -> List[Dict]:
    """
    Get curated sample songs with enhanced semantic descriptions.
    
    These 10 songs represent different genres and moods to test semantic search.
    The descriptions are crafted to work well with natural language queries like:
    - "happy dance music" → Dancing Queen
    - "sad piano ballad" → Yesterday, Imagine
    - "energetic rock workout" → Smells Like Teen Spirit, Sweet Child O Mine
    
    Returns:
        List[Dict]: List of song dictionaries with metadata
    """
    return [
        {
            'song_id': '1',
            'song_name': 'Bohemian Rhapsody',
            'band': 'Queen',
            'description': 'Bohemian Rhapsody by Queen - artist: Queen - song: Bohemian Rhapsody - rock, epic, operatic'
        },
        {
            'song_id': '2', 
            'song_name': 'Hotel California',
            'band': 'Eagles',
            'description': 'Hotel California by Eagles - artist: Eagles - song: Hotel California - rock, classic, mysterious'
        },
        {
            'song_id': '3',
            'song_name': 'Imagine',
            'band': 'John Lennon',
            'description': 'Imagine by John Lennon - artist: John Lennon - song: Imagine - peaceful, hopeful, ballad'
        },
        {
            'song_id': '4',
            'song_name': 'Stairway to Heaven',
            'band': 'Led Zeppelin',
            'description': 'Stairway to Heaven by Led Zeppelin - artist: Led Zeppelin - song: Stairway to Heaven - rock, epic, spiritual'
        },
        {
            'song_id': '5',
            'song_name': 'Purple Rain',
            'band': 'Prince',
            'description': 'Purple Rain by Prince - artist: Prince - song: Purple Rain - ballad, emotional, rainy day music'
        },
        {
            'song_id': '6',
            'song_name': 'Sweet Child O Mine',
            'band': 'Guns N Roses',
            'description': 'Sweet Child O Mine by Guns N Roses - artist: Guns N Roses - song: Sweet Child O Mine - rock, upbeat, energetic'
        },
        {
            'song_id': '7',
            'song_name': 'Yesterday',
            'band': 'The Beatles',
            'description': 'Yesterday by The Beatles - artist: The Beatles - song: Yesterday - sad, ballad, melancholic'
        },
        {
            'song_id': '8',
            'song_name': 'Dancing Queen',
            'band': 'ABBA',
            'description': 'Dancing Queen by ABBA - artist: ABBA - song: Dancing Queen - dance, party, upbeat, disco'
        },
        {
            'song_id': '9',
            'song_name': 'The Sound of Silence',
            'band': 'Simon and Garfunkel',
            'description': 'The Sound of Silence by Simon and Garfunkel - artist: Simon and Garfunkel - song: The Sound of Silence - melancholic, contemplative, folk'
        },
        {
            'song_id': '10',
            'song_name': 'Smells Like Teen Spirit',
            'band': 'Nirvana',
            'description': 'Smells Like Teen Spirit by Nirvana - artist: Nirvana - song: Smells Like Teen Spirit - rock, grunge, energetic, workout music'
        }
    ]

def get_legacy_sample_songs() -> List[Dict]:
    """
    Get legacy sample songs with basic descriptions (for backward compatibility).
    
    Returns:
        List[Dict]: List of song dictionaries with basic descriptions
    """
    return [
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
