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

2. **Backend** (`app.py`, `src/`):
   - Flask web server with async database operations
   - Modular architecture with separated concerns
   - Sentence transformer integration for embeddings
   - Vector similarity search with pgvector

3. **Database** (PostgreSQL + pgvector):
   - 768-dimensional embeddings using all-mpnet-base-v2 model
   - Enhanced descriptions for better artist and song matching
   - Optimized vector indexes for fast similarity search
   - Kaggle music dataset with rich metadata

## Code Architecture

### Modular Structure

The application uses a clean, modular architecture with separation of concerns:

```
src/
├── config.py                 # Configuration management
├── recommendation_engine.py  # Core semantic search logic
├── database_setup.py        # Database orchestrator
└── database/                # Database package
    ├── schema_manager.py    # Schema creation & management
    ├── data_loader.py       # External data loading
    ├── embedding_processor.py # Embedding operations
    └── sample_data.py       # Sample data definitions
```

### Component Responsibilities

- **`config.py`**: Environment variables and application settings
- **`recommendation_engine.py`**: Semantic similarity search and recommendations
- **`database_setup.py`**: Orchestrates database initialization using specialized modules
- **`database/schema_manager.py`**: Creates tables, indexes, and manages schema
- **`database/data_loader.py`**: Loads and processes Kaggle datasets
- **`database/embedding_processor.py`**: Generates embeddings and handles batch insertion
- **`database/sample_data.py`**: Provides curated sample music data


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

## Development Workflow

### Working with the Modular Structure

The codebase is organized into focused modules for better development experience:

```bash
# Import the main orchestrator
from src import DatabaseSetup

# Import specific database components
from src.database import SchemaManager, DataLoader, EmbeddingProcessor

# Import configuration
from src import Config
```

### Common Development Tasks

1. **Adding new data sources**: Extend `DataLoader` class
2. **Modifying database schema**: Update `SchemaManager` class  
3. **Changing embedding logic**: Modify `EmbeddingProcessor` class
4. **Adding sample data**: Update `sample_data.py`
5. **Configuration changes**: Edit `config.py`

### Testing Individual Components

```python
# Test schema management
schema_manager = SchemaManager(vector_dimensions=768)
await schema_manager.create_database_schema(connection_pool)

# Test data loading
data_loader = DataLoader(max_songs=1000)
songs = data_loader.load_kaggle_dataset()

# Test embedding processing
embedding_processor = EmbeddingProcessor(model)
embeddings = embedding_processor.generate_embeddings(descriptions)
```

## Search Capabilities

### Enhanced Description Generation

The system creates rich, semantic descriptions to improve search accuracy:

```python
# Example enhanced description format:
"Money, Money, Money (ABBA cover) by ReinXeed - artist: ReinXeed - song: Money, Money, Money (ABBA cover)"
```

This approach ensures:
- **Artist searches** like "ReinXeed" find all songs by that artist
- **Song searches** like "Money Money Money" find the specific song
- **Genre/mood searches** like "sad songs" find emotionally relevant music

### Search Examples

```bash
# Artist-based search
curl -X POST /api/recommend -d '{"query": "ABBA", "limit": 5}'

# Song-based search  
curl -X POST /api/recommend -d '{"query": "Dancing Queen", "limit": 5}'

# Mood-based search
curl -X POST /api/recommend -d '{"query": "sad ballads", "limit": 5}'

# Genre-based search
curl -X POST /api/recommend -d '{"query": "rock music", "limit": 5}'
```

## Understanding the Code

This project demonstrates production-ready pgvector implementation with:

- **Semantic Search**: Convert text queries to 768D embeddings using sentence-transformers
- **Vector Storage**: Store embeddings in PostgreSQL with pgvector extension
- **Similarity Search**: Use cosine distance for finding similar songs
- **Production Patterns**: Async operations, connection pooling, error handling

**For detailed code explanations, SQL queries, and implementation patterns, see [guide.md](guide.md#code-walkthrough).**

## Customization Guide

### Adding New Data Sources

1. **Create a new data loader in `src/database/data_loader.py`**:
   ```python
   def load_custom_dataset(self):
       # Your implementation here
       return songs_list
   ```

2. **Enhanced descriptions are automatically generated**:
   ```python
   # The DataLoader creates enhanced descriptions automatically:
   description = f"{song_name} by {artist_name} - artist: {artist_name} - song: {song_name}"
   ```

3. **Embeddings are handled by EmbeddingProcessor**:
   ```python
   # EmbeddingProcessor automatically handles:
   embeddings = self.embedding_processor.generate_embeddings(descriptions)
   ```

4. **Schema updates via SchemaManager**:
   ```python
   # Update vector dimensions in SchemaManager
   schema_manager = SchemaManager(vector_dimensions=new_dimension)
   ```

### Changing the ML Model

```python
# Current model: all-mpnet-base-v2 (768 dimensions)
# To change model, update src/config.py:
SENTENCE_TRANSFORMER_MODEL = 'your-model-name'

# Important: Update vector dimensions in database schema
# Current schema uses VECTOR(768) for all-mpnet-base-v2
# Different models may require different dimensions
```

### UI Customization

- **Styling**: Edit `static/css/style.css`
- **Functionality**: Modify `static/js/app.js`
- **Layout**: Update `templates/index.html`

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

