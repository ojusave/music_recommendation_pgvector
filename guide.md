# pgvector Music Recommendations

Semantic search demo using **pgvector** with PostgreSQL. Live deployment showcasing vector similarity search and AI embeddings.

## Quick Deploy

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

1. Fork repo → Connect to Render → Add PostgreSQL + pgvector → Set DATABASE_URL
2. App auto-initializes schema and sample data

**Live Demo:** [music-recommendations-hz3i.onrender.com](https://music-recommendations-hz3i.onrender.com)

## What This Demonstrates

- **Semantic search** with pgvector cosine similarity
- **Vector embeddings** using sentence-transformers (768D)
- **Production indexing** with IVFFlat
- **Real-time API** with sub-second response times

## Architecture

```
src/
├── recommendation_engine.py  # Core pgvector search
├── database_setup.py        # Schema + data loading
└── database/               # Modular components
```

**Flow:** Query → AI embedding → pgvector search → ranked results

## Key Implementation

### Database Schema
```sql
CREATE EXTENSION vector;

CREATE TABLE songs (
    song_name TEXT,
    band TEXT,
    description TEXT,
    embedding VECTOR(768)
);

CREATE INDEX ON songs USING ivfflat (embedding vector_cosine_ops);
```

### Embedding Generation
```python
def generate_embeddings(self, descriptions):
    embeddings = self.model.encode(descriptions)
    # Normalize for cosine similarity
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
```

### Similarity Search
```python
search_query = """
SELECT song_name, band,
       (embedding <-> $1::vector) as distance,
       ROUND((1.0 - (embedding <-> $1::vector) / 2.0) * 100.0, 1) as similarity
FROM songs 
ORDER BY embedding <-> $1::vector ASC
LIMIT $2
"""
```

### API Endpoint
```python
@app.route('/api/recommend', methods=['POST'])
def recommend():
    query = request.json['query']
    recommendations = engine.get_recommendations(query, limit=5)
    return jsonify({'recommendations': recommendations})
```

## Try It

**Example searches:**
- `"ABBA"` → All ABBA songs
- `"sad piano music"` → Melancholic piano pieces
- `"upbeat workout"` → High-energy tracks

**API:**
```bash
curl -X POST /api/recommend -d '{"query": "sad ballads"}'
```

## Production Features

- **Connection pooling** with asyncpg
- **Health checks** at `/health`
- **Auto-initialization** of schema/data
- **Error handling** and monitoring
- **Horizontal scaling** ready

Built with Flask, PostgreSQL 15+, sentence-transformers, deployed on Render with pgvector extension.