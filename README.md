# Music Recommendation App with pgvector

A modern web application that demonstrates semantic music search using **pgvector** with **Render's PostgreSQL**. Users can describe music in natural language and get intelligent recommendations based on semantic similarity.

## Live Demo

[**Try the live app**](https://music-recommendations.onrender.com) *(Deploy to get your URL)*

## Features

- **Natural Language Search**: "upbeat rock music for working out" leads to relevant song recommendations
- **Semantic Similarity**: Uses sentence transformers and vector embeddings for intelligent matching
- **Real-time Results**: Fast similarity search with pgvector's optimized indexing
- **Music Service Integration**: Direct links to YouTube and Spotify
- **Responsive Design**: Works beautifully on desktop and mobile
- **Production Ready**: Configured for easy deployment on Render

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Flask API     │    │  PostgreSQL     │
│                 │    │                 │    │  + pgvector     │
│ • HTML/CSS/JS   │◄──►│ • Async Routes  │◄──►│ • Vector Search │
│ • Search UI     │    │ • ML Pipeline   │    │ • Cosine Sim    │
│ • Results       │    │ • Error Handle  │    │ • Indexing      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key Components

1. **Frontend** (`templates/index.html`, `static/`):
   - Clean, modern interface for music discovery
   - Real-time search with loading states
   - Responsive design with example queries

2. **Backend** (`app.py`):
   - Flask web server with async database operations
   - Sentence transformer integration for embeddings
   - Vector similarity search with pgvector

3. **Database** (PostgreSQL + pgvector):
   - 384-dimensional embeddings for 10,000+ songs
   - Optimized vector indexes for fast similarity search
   - Kaggle music dataset with rich metadata

## Quick Deploy to Render

### Option 1: One-Click Deploy
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

### Option 2: Manual Deploy

1. **Fork this repository**

2. **Create a new Web Service on Render**:
   - Connect your GitHub repository
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn --bind 0.0.0.0:$PORT app:app --workers 2`

3. **Create PostgreSQL Database**:
   - Add a new PostgreSQL database
   - Enable pgvector extension
   - Note the connection string

4. **Set Environment Variables**:
   ```
   DATABASE_URL=your_postgres_connection_string
   FLASK_ENV=production
   ```

5. **Load Sample Data**:
   ```bash
   python process_data.py  # Run once to populate database
   ```

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
   python process_data.py
   ```

6. **Run the application**:
   ```bash
   python app.py
   ```

7. **Open in browser**: http://localhost:5000

## Understanding the Code

### Vector Embeddings Pipeline

```python
# 1. Convert text to embeddings
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
query_embedding = model.encode("upbeat rock music")

# 2. Store in PostgreSQL with pgvector
CREATE TABLE songs (
    song_id SERIAL PRIMARY KEY,
    song_name VARCHAR(500),
    band VARCHAR(300),
    embedding VECTOR(384)  -- 384 dimensions
);

# 3. Perform similarity search
SELECT song_name, band, 
       (embedding <-> $1::vector) as distance
FROM songs 
ORDER BY embedding <-> $1::vector
LIMIT 5;
```

### Key Implementation Details

1. **Vector Normalization**: 
   - Consistent normalization between query and stored vectors
   - Critical for accurate cosine similarity

2. **Database Indexing**:
   ```sql
   CREATE INDEX songs_embedding_idx 
   ON songs USING ivfflat (embedding vector_cosine_ops);
   ```

3. **Similarity Scoring**:
   - Converts raw cosine distances to 0-100% similarity scores
   - Better user experience than raw distance values

4. **Async Database Operations**:
   - Uses `asyncpg` for non-blocking database queries
   - Connection pooling for production performance

## Customization Guide

### Adding New Data Sources

1. **Modify `process_data.py`**:
   ```python
   # Add your data loading logic
   def load_custom_dataset():
       # Your implementation here
       return songs_dataframe
   ```

2. **Update embedding generation**:
   ```python
   # Generate embeddings for your data
   embeddings = model.encode(songs_dataframe['description'])
   ```

3. **Adjust vector dimensions if needed**:
   ```sql
   -- Update table schema
   ALTER TABLE songs ALTER COLUMN embedding TYPE VECTOR(new_dimension);
   ```

### Changing the ML Model

```python
# In app.py, replace with your preferred model
model = SentenceTransformer('your-model-name')

# Update vector dimensions in database schema
# Recreate indexes with new dimensions
```

### UI Customization

- **Styling**: Edit `static/css/style.css`
- **Functionality**: Modify `static/js/app.js`
- **Layout**: Update `templates/index.html`

## Production Considerations

### Performance Optimization

1. **Database Tuning**:
   ```sql
   -- Adjust index parameters based on data size
   CREATE INDEX songs_embedding_idx 
   ON songs USING ivfflat (embedding vector_cosine_ops) 
   WITH (lists = 1000);  -- Increase for larger datasets
   ```

2. **Connection Pooling**:
   ```python
   # Adjust pool size based on traffic
   pool = await asyncpg.create_pool(
       DATABASE_URL, 
       min_size=5, 
       max_size=20
   )
   ```

3. **Caching**: Consider adding Redis for frequently searched queries

### Security

- Use environment variables for all secrets
- Enable CORS protection for production
- Implement rate limiting for API endpoints
- Validate and sanitize all user inputs

### Monitoring

- Add application logging with structured logs
- Monitor database performance and query times
- Set up health check endpoints
- Track user engagement metrics

## 📚 Learning Resources

### pgvector Documentation
- [pgvector GitHub](https://github.com/pgvector/pgvector)
- [PostgreSQL Vector Operations](https://github.com/pgvector/pgvector#vector-operations)

### Sentence Transformers
- [SentenceTransformers Documentation](https://www.sbert.net/)
- [Model Hub](https://huggingface.co/sentence-transformers)

### Render Platform
- [Render Docs](https://render.com/docs)
- [PostgreSQL on Render](https://render.com/docs/databases)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Render Support**: [Render Help](https://render.com/docs/support)

---

**Built for the Render community**

*This demo showcases the power of combining modern ML techniques with PostgreSQL's vector capabilities on Render's platform.*
