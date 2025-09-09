# Music Recommendation App

## Table of Contents
- [What is this?](#what-is-this)
- [Quick Start](#quick-start)
- [Live Demo](#live-demo)
- [Local Development](#local-development)
- [API Documentation](#api-documentation)
- [Troubleshooting](#troubleshooting)

## What is this?
A web app that lets you search for music using natural language (like "sad piano songs") instead of exact titles. It uses AI to understand what you mean and finds similar music.

## How it works
1. You type "upbeat rock songs" 
2. AI converts your words into numbers (vectors)
3. Database finds songs with similar numbers
4. You get recommendations!

**Tech Stack:** Flask + PostgreSQL + pgvector + Sentence Transformers

## Quick Start

**One-Click Deploy:**
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

**What's a vector?** Think of it as a list of numbers that represents the "meaning" of text. Similar meanings have similar numbers.

**Example:**
- "happy song" → [0.1, 0.8, 0.2, ...]
- "joyful music" → [0.2, 0.7, 0.3, ...] (similar numbers!)
- "sad song" → [0.9, 0.1, 0.8, ...] (different numbers)

**What's pgvector?** A PostgreSQL extension that can store these number lists and find similar ones quickly.

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
│ "Moonlight      │    │ Finds similar    │    │ Stores vectors  │
│  Sonata"        │    │ vectors          │    │ + song info     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Quick Start (5 minutes)

**Option 1: One-Click Deploy (Recommended for beginners)**
1. Click this button: [![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)
2. Wait 2-3 minutes for deployment
3. Done! Your app is live.

**Option 2: Manual (If you want to understand each step)**
[See Manual Render Deployment section below]

## Live Demo

**Working Example**: [https://music-recommendations-hz3i.onrender.com/](https://music-recommendations-hz3i.onrender.com/)

Try the live demo with natural language queries like:
- "sad piano music"
- "upbeat rock songs" 
- "electronic dance music"
- "acoustic folk ballads"

## Try These Searches

- **"happy dance music"** → "Happy" by Pomplamoose (53.6%)
- **"sad piano music"** → "Lonely Day (piano cover)" by System Of A Down (55.6%)
- **"upbeat rock songs"** → "Rock It" by Sub Focus (54.2%)

*Results show similarity scores from the live demo*

## Prerequisites
- Python 3.9+
- PostgreSQL 12+ with pgvector extension

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