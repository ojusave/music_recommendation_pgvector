# Music Recommendation App with pgvector

A modern web application that demonstrates semantic music search using **pgvector** with **PostgreSQL**. Users can describe music in natural language and get intelligent recommendations based on semantic similarity.

**New to pgvector?** See [guide.md](guide.md) for a comprehensive tutorial on pgvector fundamentals, concepts, and implementation patterns.

## Live Demo

[**Try the live app**](https://music-recommendations-hz3i.onrender.com)

## Features

- **Natural Language Search**: "upbeat rock music for working out" or "ABBA songs" leads to relevant recommendations
- **Artist-Based Search**: Search by artist name (e.g., "ReinXeed") returns all songs by that artist
- **Semantic Similarity**: Uses all-mpnet-base-v2 sentence transformers and 768-dimensional vector embeddings
- **Real-time Results**: Fast similarity search with pgvector's optimized indexing
- **Music Service Integration**: Direct links to YouTube and Spotify
- **Production Ready**: Configured for easy deployment

## How It Works

**Simple concept:** Type "sad piano music" → Get relevant song recommendations

**Technology stack:** Flask + PostgreSQL + pgvector + AI embeddings

**For detailed architecture, code explanations, and pgvector concepts, see [guide.md](guide.md#project-architecture-overview).**


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

### Prerequisites
- Python 3.11+
- PostgreSQL with pgvector extension
- Git

### Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd music-recommendation-app
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up PostgreSQL with pgvector**:
   ```sql
   CREATE DATABASE music_recommendations;
   CREATE EXTENSION vector;
   ```

4. **Configure environment**:
   ```bash
   export DATABASE_URL="postgresql://user:pass@localhost/music_recommendations"
   ```

5. **Load sample data**:
   ```bash
   # Data is loaded automatically - no manual step needed
   ```

6. **Run the application**:
   ```bash
   python app.py
   ```

7. **Open in browser**: http://localhost:5000

## Development

**Quick start:** Follow setup steps above, then modify code as needed.

**For detailed development workflows, component testing, and customization guides, see [guide.md](guide.md#code-walkthrough).**

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

## Understanding the Code

This project demonstrates production-ready pgvector implementation with:

- **Semantic Search**: Convert text queries to 768D embeddings using sentence-transformers
- **Vector Storage**: Store embeddings in PostgreSQL with pgvector extension
- **Similarity Search**: Use cosine distance for finding similar songs
- **Production Patterns**: Async operations, connection pooling, error handling

**For detailed code explanations, SQL queries, and implementation patterns, see [guide.md](guide.md#code-walkthrough).**

## Customization

**Want to customize this app?**

- **Change AI model**: Update `SENTENCE_TRANSFORMER_MODEL` in config
- **Add data sources**: Extend the `DataLoader` class  
- **Modify UI**: Edit files in `templates/` and `static/`

**For detailed customization guides and best practices, see [guide.md](guide.md#best-practices).**

## Production Considerations

This app is production-ready with connection pooling, error handling, and health checks built-in.

**For comprehensive production guidance including:**
- Environment configuration and .env setup
- Database tuning and indexing strategies  
- Performance optimization patterns
- Security best practices
- Monitoring and troubleshooting

**See [guide.md](guide.md#production-considerations) for detailed production deployment patterns.**

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

## Troubleshooting

**Common Issues:**
- **Port 5000 in use**: Change port with `export PORT=5001`
- **Database connection**: Verify `DATABASE_URL` environment variable  
- **Model loading**: First run downloads model (may be slow)
- **Import errors**: Use `from src.database import SchemaManager`

**For comprehensive troubleshooting including:**
- Vector dimension mismatch errors
- pgvector extension issues
- Render deployment problems
- Database schema debugging

**See [guide.md](guide.md#common-patterns-and-pitfalls) for detailed troubleshooting guides.**

