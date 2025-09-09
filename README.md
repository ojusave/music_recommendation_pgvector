# Music Recommendation App

Semantic music search using **pgvector** with PostgreSQL and sentence transformers. Search with natural language queries like "sad piano music" to get intelligent recommendations.

## Features

- **Semantic Search**: Natural language queries using sentence transformers
- **Vector Similarity**: pgvector extension for fast similarity search  
- **Auto-Setup**: Database and sample data initialize automatically
- **Music Links**: Direct YouTube and Spotify search links
- **REST API**: JSON API for programmatic access
- **Memory Optimized**: Uses lightweight model for deployment efficiency

## Quick Deploy

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

1. Connect to Render → Add PostgreSQL + pgvector → Set `DATABASE_URL`
2. App auto-initializes with sample music data

## Local Development

```bash
pip install -r requirements.txt
export DATABASE_URL="postgresql://user:pass@localhost/music_db"
# Create database with pgvector extension
createdb music_db
psql music_db -c "CREATE EXTENSION vector;"
python app.py
```

## API Usage

```bash
curl -X POST /api/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "upbeat rock songs", "limit": 5}'
```

**Health Check:** `GET /health`

## Understanding Similarity Scores

The app uses **cosine distance** for vector similarity, which affects the percentage scores:

- **Score Formula**: `(1.0 - distance/2.0) * 100`
- **Distance Range**: 0 (identical) to 2 (opposite)
- **Typical Scores**: 30-60% for semantic matches, 80%+ for exact matches

**Why scores might be lower than expected:**
- Sentence transformer models aren't specifically trained on music data
- Sample song descriptions may lack semantic richness
- Different query phrasing creates different embeddings ("disney" vs "walt disney")

**Note:** Scores below 50% are normal for semantic search - focus on relative ranking rather than absolute percentages.

