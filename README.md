# Music Recommendation App with pgvector

A modern web application that demonstrates semantic music search using **pgvector** with **PostgreSQL**. Users can describe music in natural language and get intelligent recommendations based on semantic similarity.

**New to pgvector?** See [guide.md](guide.md) for a comprehensive tutorial on pgvector fundamentals, concepts, and implementation patterns.

## Live Demo

[**Try the live app**](https://music-recommendations-hz3i.onrender.com)

## What This App Does

Type **"sad piano music"** → Get intelligent song recommendations

**Key features:**
- Natural language search ("upbeat workout music")
- Artist and song search ("ABBA", "Dancing Queen")  
- Direct links to YouTube and Spotify
- Production-ready deployment

**How it works:** AI converts your search into numbers, PostgreSQL finds similar songs using pgvector.

**Want to understand the technology?** See [guide.md](guide.md) for comprehensive pgvector tutorials and implementation details.


## Quick Deploy

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

**Simple Steps:**
1. Fork this repository
2. Connect to Render and deploy
3. Add PostgreSQL database with pgvector
4. Set `DATABASE_URL` environment variable
5. App auto-initializes on first run

**For detailed deployment instructions including pgvector setup, environment configuration, and troubleshooting, see [guide.md](guide.md#deployment-on-render).**

## Local Development

**Prerequisites:** Python 3.11+, PostgreSQL with pgvector extension

**Setup:**
```bash
# 1. Clone and install
git clone <repo-url>
cd music-recommendation-app
pip install -r requirements.txt

# 2. Setup database
createdb music_recommendations
psql music_recommendations -c "CREATE EXTENSION vector;"

# 3. Configure and run
export DATABASE_URL="postgresql://user:pass@localhost/music_recommendations"
python app.py
```

**For detailed setup, environment configuration, and development workflows, see [guide.md](guide.md#database-setup).**

## Search Examples

**Try these searches:**
- "ABBA" → Find all ABBA songs
- "sad piano music" → Find melancholic piano pieces  
- "upbeat workout songs" → Find energetic music
- "Dancing Queen" → Find the specific song

**API Usage:**
```bash
curl -X POST /api/recommend -d '{"query": "sad ballads", "limit": 5}'
```

## Code Overview

This is a production-ready pgvector example with Flask, PostgreSQL, and AI embeddings.

**For detailed code explanations, architecture, and implementation patterns, see [guide.md](guide.md#understanding-the-codebase).**

## Customization & Production

**This app is production-ready** with built-in connection pooling, error handling, and health checks.

**For customization, production deployment, performance tuning, and best practices, see [guide.md](guide.md#production-deployment).**

## Learning Resources

### pgvector Documentation
- [pgvector GitHub](https://github.com/pgvector/pgvector)
- [PostgreSQL Vector Operations](https://github.com/pgvector/pgvector#vector-operations)

### Sentence Transformers
- [SentenceTransformers Documentation](https://www.sbert.net/)
- [Model Hub](https://huggingface.co/sentence-transformers)

### Render Platform
- [Render Docs](https://render.com/docs)
- [PostgreSQL on Render](https://render.com/docs/databases)

## Need Help?

**Quick fixes:** Port issues → change PORT env var | Database issues → check DATABASE_URL | Slow startup → model downloading

**For comprehensive troubleshooting, deployment issues, and debugging guides, see [guide.md](guide.md#troubleshooting).**

