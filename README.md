# Music Recommendation App

## Table of Contents
- [What is this?](#what-is-this)
- [Quick Start](#quick-start)
- [How the Search Works](#how-the-search-works)
- [Visual Architecture](#visual-architecture)
- [Live Demo](#live-demo)
- [Try These Searches](#try-these-searches)
- [Prerequisites](#prerequisites)
- [Local Development](#local-development)
- [API Documentation](#api-documentation)
- [Troubleshooting](#troubleshooting)

## What is this?
A web app that lets you search for music using natural language (like "sad piano songs") instead of exact titles. It uses AI to understand what you mean and finds similar music.

**Tech Stack:** Flask + PostgreSQL + pgvector + Sentence Transformers

## Quick Start 

**One-Click Deploy:** 

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

*Wait 2-3 minutes and your app is live!*

**What's a vector?** 

Think of it as a list of numbers that represents the "meaning" of text. Similar meanings have similar numbers.

**Example:**
- "happy song" → [0.1, 0.8, 0.2, ...]
- "joyful music" → [0.2, 0.7, 0.3, ...] (similar numbers!)
- "sad song" → [0.9, 0.1, 0.8, ...] (different numbers)

**What's pgvector?** A PostgreSQL extension that can store these number lists and find similar ones quickly.

## How the Search Works

1. **Text to Vector**: Your query "sad piano music" gets converted to a 384-dimensional vector using a pre-trained sentence transformer model
2. **Vector Normalization**: The vector is normalized to unit length for proper cosine similarity calculation
3. **Database Search**: pgvector compares your query vector against all song vectors using cosine distance (`<->` operator)
4. **Similarity Scoring**: Raw distances (0-2) are converted to percentage scores: `(1.0 - distance/2.0) * 100`
5. **Ranking**: Songs are ranked by similarity score and returned as recommendations

**Why this works**: The sentence transformer model was trained on millions of text pairs, learning that "sad" and "melancholic" have similar meanings, "piano" and "ballad" often go together, etc.

## pgvector Optimizations for Production

### Performance Improvements (NEW!)

This app now uses advanced **pgvector features** for optimal performance on Render's starter tier (0.5 CPU, 512MB memory):

#### 1. **Half-Precision Vectors (50% Memory Savings)**
```sql
-- Before: Standard vectors (1.5KB per song)
embedding VECTOR(384)

-- After: Half-precision vectors (768 bytes per song) 
embedding HALFVEC(384)
```
**Impact**: 50,000 songs now use 37.5MB instead of 75MB!

#### 2. **Native Cosine Distance**
```sql
-- Before: L2 distance on normalized vectors
ORDER BY embedding <-> $1::vector ASC

-- After: pgvector's native cosine distance (more efficient)
ORDER BY embedding <=> l2_normalize($1::halfvec) ASC
```

#### 3. **Database-Level Vector Operations**
```sql
-- Vector normalization now happens in PostgreSQL (faster)
SELECT embedding <=> l2_normalize($1::halfvec) as cosine_distance
```

#### 4. **Optimized Indexing**
```sql
-- Dynamic index sizing based on dataset size
CREATE INDEX songs_embedding_idx 
ON songs USING ivfflat (embedding halfvec_cosine_ops) 
WITH (lists = SQRT(num_songs));
```

#### 5. **Streaming Batch Processing**
- Large datasets processed in 50-song chunks
- Automatic garbage collection after each chunk
- Memory usage stays constant regardless of dataset size

### Memory Usage Comparison
| Dataset Size | Before (vector) | After (halfvec) | Memory Saved |
|--------------|----------------|-----------------|--------------|
| 10K songs    | 15MB           | 7.5MB          | 50%          |
| 50K songs    | 75MB           | 37.5MB         | 50%          |
| 100K songs   | 150MB          | 75MB           | 50%          |

## Model Selection & Tradeoffs

### Current Model: `paraphrase-MiniLM-L3-v2`

**What we know:**
- **384-dimensional vectors**: As mentioned in the search process above
- **Pre-trained sentence transformer**: Trained on text pairs to understand semantic similarity
- **Memory optimized**: The README suggests this as a "smaller model" for memory-constrained deployments
- **Configurable**: Can be changed via `SENTENCE_TRANSFORMER_MODEL` environment variable

**General Tradeoffs in Sentence Transformer Models:**
- **Larger models**: Typically more accurate but require more memory and compute
- **Smaller models**: Faster and more memory-efficient but may be less accurate
- **Specialized models**: Some are optimized for specific domains or languages

**Alternative Models:**
The app supports other sentence transformer models via environment variables. Common alternatives include models from the sentence-transformers library, but specific performance characteristics would need to be tested for your use case.

**Considerations:**
- Model choice depends on your deployment constraints (memory, speed requirements)
- Different models may perform differently on music-related queries
- Test with your specific queries to determine the best model for your needs

**Important: Changing Models Requires Re-embedding**

If you change the sentence transformer model, you **must** regenerate all embeddings in your database because:
- Different models produce incompatible vector representations
- A song embedded with `paraphrase-MiniLM-L3-v2` cannot be compared with a query embedded using `all-MiniLM-L6-v2`
- Vector dimensions may also change between models

**Steps to change models:**
1. Update `SENTENCE_TRANSFORMER_MODEL` environment variable
2. Clear existing embeddings from database
3. Re-run the embedding process for all songs
4. This may take significant time for large datasets

## Visual Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Your Query    │───▶│   AI Model       │───▶│   Vector        │
│ "sad piano"     │    │ Converts to      │    │ [0.2, 0.8, ...] │
└─────────────────┘    │ numbers          │    └─────────────────┘
                       └──────────────────┘              │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Results       │◀───│   pgvector       │◀───│   Database      │
│ "Lonely Day     │    │ Finds similar    │    │ Stores vectors  │
│ (piano cover)"  │    │ vectors          │    │ + song info     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Live Demo

**Example**: [https://music-recommendations-hz3i.onrender.com/](https://music-recommendations-hz3i.onrender.com/)

Try the live demo with natural language queries like:
- "sad piano music"
- "upbeat rock songs" 
- "electronic dance music"
- "acoustic folk ballads"

## Try These Searches

- **"happy dance music"** → "Happy" by Pomplamoose (53.6%)
- **"sad piano music"** → "Lonely Day (piano cover)" by System Of A Down (55.6%)
- **"upbeat rock songs"** → "Rock It" by Sub Focus (54.2%)


## Prerequisites
- Python 3.9+
- PostgreSQL 15+ with pgvector 0.7.0+ extension
  - **Required for halfvec support** (50% memory savings)
  - **Older pgvector versions** will automatically fall back to standard vectors

## Local Development

1. **Clone and Install Dependencies**
```bash
mkdir <folder_name>
cd <folder_name>
git clone https://github.com/ojusave/music_recommendation_pgvector.git
pip install -r requirements.txt
```

2. **Database Setup**
```bash
# Create database
createdb music_recommendations

# Connect and enable pgvector extension
psql music_recommendations -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Set database URL
export DATABASE_URL="postgresql://username:password@localhost:5432/music_recommendations"
```

3. **Configure Environment Variables**
```bash
# Required
export DATABASE_URL="postgresql://username:password@localhost:5432/music_recommendations"

# Optional optimizations
export SENTENCE_TRANSFORMER_MODEL="paraphrase-MiniLM-L3-v2"
export OPTIMIZE_FOR_MEMORY="false"  # Set to "true" for 512MB deployments
export LOG_LEVEL="INFO"
export FLASK_DEBUG="true"  # For development only
export SECRET_KEY="<generate-random-secret>"
export FLASK_ENV="production"  # For production deployment

# pgvector Optimizations (NEW!)
export USE_HALFVEC="true"           # Enable half-precision vectors (50% memory savings)
export MAX_KAGGLE_SONGS="50000"     # Maximum songs to load from Kaggle dataset
export EMBEDDING_BATCH_SIZE="50"    # Batch size for memory-efficient processing
```

4. **Run Application**
```bash
python app.py
```

The app will start on http://localhost:5000 with sample data loaded automatically.

## API Documentation

### Endpoints

#### GET /health
Health check endpoint for monitoring.

**Example:**
```bash
curl https://music-recommendations-hz3i.onrender.com/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "music-recommendations",
  "message": "App is running"
}
```

#### POST /api/recommend
Get music recommendations based on natural language query.

**Example:**
```bash
curl -X POST https://music-recommendations-hz3i.onrender.com/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "upbeat rock songs", "limit": 5}'
```

**Request:**
```json
{
  "query": "upbeat rock songs",
  "limit": 5
}
```

**Response:**
```json
{
  "success": true,
  "recommendations": [
    {
      "artist": "Sub Focus",
      "band": "Sub Focus",
      "raw_distance": 0.9157,
      "similarity_score": 54.2,
      "song_name": "Rock It",
      "spotify_url": "https://open.spotify.com/search/Rock%20It%20Sub%20Focus",
      "youtube_url": "https://www.youtube.com/results?search_query=Rock%20It%20Sub%20Focus"
    },
    {
      "artist": "Funkdoobiest",
      "band": "Funkdoobiest",
      "raw_distance": 0.9419,
      "similarity_score": 52.9,
      "song_name": "Rock On",
      "spotify_url": "https://open.spotify.com/search/Rock%20On%20Funkdoobiest",
      "youtube_url": "https://www.youtube.com/results?search_query=Rock%20On%20Funkdoobiest"
    }
  ],
  "query": "upbeat rock songs",
  "count": 2
}
```

#### GET /api/status
Get database statistics and health information.

**Example:**
```bash
curl http://localhost:5000/api/status
```

**Response:**
```json
{
  "total_songs": 10000,
  "sample_artists": ["Sub Focus", "Funkdoobiest", "Motorhead", "System Of A Down", "Classical New Age Piano Music"],
  "status": "healthy"
}
```



## Troubleshooting

**Database connection failed:**
- Check `DATABASE_URL` is set
- Enable pgvector: `psql $DATABASE_URL -c "CREATE EXTENSION IF NOT EXISTS vector;"`

**Out of memory errors:**
- Set `OPTIMIZE_FOR_MEMORY=true`
- Use smaller model: `paraphrase-MiniLM-L3-v2`

**Low similarity scores:**
- Normal for semantic search (focus on ranking, not absolute %)
- Try different query phrasings