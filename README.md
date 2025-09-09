# Music Recommendation App

## What is this?
A web app that lets you search for music using natural language (like "sad piano songs") instead of exact titles. It uses AI to understand what you mean and finds similar music.

## How it works (Simple version)
1. You type "upbeat rock songs" 
2. AI converts your words into numbers (vectors)
3. Database finds songs with similar numbers
4. You get recommendations!

## Key Technologies (What you'll learn)
- **pgvector**: PostgreSQL extension that stores and searches AI vectors
- **Render**: Cloud platform that hosts your app and database
- **Sentence Transformers**: AI model that converts text to vectors
- **Flask**: Python web framework

## Prerequisites
- Basic Python knowledge
- Basic SQL knowledge (helpful but not required)
- No AI/ML experience needed - we'll explain!

## New to Vector Databases?

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

## Absolute Beginner Deployment (5 minutes)

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

## Try These Searches (See How It Works)

Start simple:
- "happy music" → High matches: Dancing Queen, Don't Stop Me Now
- "sad songs" → High matches: Yesterday, The Sound of Silence

Get specific:
- "upbeat rock workout music" → Sweet Child O Mine, Smells Like Teen Spirit
- "rainy day piano ballad" → Imagine, Purple Rain

Why does this work?
The AI learned that "happy" and "upbeat" mean similar things, "piano" and "ballad" go together, etc.

## Project Structure (What each file does)

```
app.py                 # Main web app (start here!)
src/
  ├── config.py        # Settings (database URL, model choice, etc.)
  ├── recommendation_engine.py  # The AI magic happens here
  └── database/        # Database stuff
      ├── schema_manager.py     # Creates tables
      ├── data_loader.py        # Loads sample music
      └── sample_data.py        # 10 example songs
render.yaml           # Tells Render how to deploy
requirements.txt      # Python packages needed
```

**New to this?** Start by reading `app.py`, then `config.py`

## Learning Path for Beginners

### Phase 1: Get it running (15 minutes)
1. Deploy with one-click
2. Try the example searches
3. Look at the results

### Phase 2: Understand the code (30 minutes)
1. Read `app.py` - see how searches work
2. Look at `sample_data.py` - see the example songs
3. Check `config.py` - understand the settings

### Phase 3: Customize it (1 hour)
1. Add your own songs to `sample_data.py`
2. Try different search phrases
3. Modify the similarity scoring

### Phase 4: Advanced (optional)
1. Connect to real music APIs
2. Add more sophisticated AI models
3. Scale to millions of songs

## Common Issues (New to Render?)

**"Database connection failed"**
- Check your DATABASE_URL is set correctly
- Make sure pgvector extension is enabled: `CREATE EXTENSION vector;`

**"Out of memory" errors**
- Set `OPTIMIZE_FOR_MEMORY=true`
- Using Render's free tier? It has 512MB RAM limit

**"Model loading takes forever"**
- First startup is slow (downloading AI model)
- Subsequent starts are faster (model is cached)

**Low similarity scores (everything shows 20-30%)**
- This is normal! Focus on ranking, not absolute percentages
- Try different search phrases

## API Examples with Live Demo

```bash
# Health check
curl https://music-recommendations-hz3i.onrender.com/health

# Get recommendations
curl -X POST https://music-recommendations-hz3i.onrender.com/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "upbeat rock songs", "limit": 5}'

# Database status
curl https://music-recommendations-hz3i.onrender.com/api/status
```

## Local Development Setup

### Prerequisites
- Python 3.9+
- PostgreSQL 12+ with pgvector extension

### Installation Steps

1. **Clone and Install Dependencies**
```bash
git clone <repository-url>
cd render-test
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
```

4. **Run Application**
```bash
python app.py
```

The application will:
- Initialize the sentence transformer model
- Create database schema automatically
- Load sample music data
- Start the Flask server on http://localhost:5000

## Manual Render Deployment

If you prefer manual deployment instead of one-click:

### Step 1: Create PostgreSQL Database
1. In Render Dashboard, create a new PostgreSQL service
2. Name it `music-recommendations-db`
3. Select the Free plan
4. Note the connection details

### Step 2: Enable pgvector Extension
Connect to your database and run:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Step 3: Create Web Service
1. Create new Web Service in Render
2. Connect your GitHub repository
3. Configure build and start commands:

**Build Command:**
```bash
pip install --no-cache-dir -r requirements.txt
```

**Start Command:**
```bash
gunicorn --bind 0.0.0.0:$PORT app:app --workers 1 --timeout 240
```

### Step 4: Environment Variables
Set these environment variables in Render:

```bash
DATABASE_URL=<your-postgres-connection-string>
SENTENCE_TRANSFORMER_MODEL=paraphrase-MiniLM-L3-v2
OPTIMIZE_FOR_MEMORY=true
FLASK_ENV=production
SECRET_KEY=<generate-random-secret>
LOG_LEVEL=INFO
```

## API Documentation

### Endpoints

#### GET /health
Health check endpoint for monitoring.

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
      "song_id": "1",
      "song_name": "Don't Stop Me Now",
      "artist": "Queen",
      "description": "Upbeat rock anthem with powerful vocals",
      "similarity_score": 85.2,
      "raw_distance": 0.148,
      "youtube_url": "https://www.youtube.com/results?search_query=Don%27t+Stop+Me+Now+Queen",
      "spotify_url": "https://open.spotify.com/search/Don%27t+Stop+Me+Now+Queen"
    }
  ],
  "query": "upbeat rock songs",
  "count": 1
}
```

#### GET /api/status
Get database statistics and health information.

**Response:**
```json
{
  "total_songs": 1000,
  "sample_artists": ["Queen", "Beatles", "Led Zeppelin"],
  "status": "healthy"
}
```

## Understanding Similarity Scores

The application uses cosine distance for vector similarity with the following characteristics:

### Score Calculation
```python
# Raw distance from pgvector (0 = identical, 2 = opposite)
raw_distance = embedding <-> query_vector

# Converted to percentage score
similarity_score = (1.0 - raw_distance / 2.0) * 100.0
```

### Score Interpretation
- **80%+**: Very high similarity (near-exact matches)
- **60-80%**: High similarity (strong semantic match)
- **40-60%**: Moderate similarity (related concepts)
- **20-40%**: Low similarity (loosely related)
- **<20%**: Very low similarity (unrelated)

### Why Scores May Be Lower Than Expected
1. **Model Limitations**: Sentence transformers aren't specifically trained on music data
2. **Description Quality**: Sample data may lack rich semantic descriptions
3. **Query Phrasing**: Different phrasings create different embeddings ("disney" vs "walt disney")
4. **Domain Gap**: General language models may not capture music-specific semantics

**Note**: Focus on relative ranking rather than absolute percentages. Lower scores are normal for semantic search.

---

## Advanced Technical Details

### Core Architecture

This application implements a modern semantic search architecture with the following key components:

- **Semantic Search Engine**: Uses sentence transformers to convert natural language queries into vector embeddings
- **Vector Database**: PostgreSQL with pgvector extension for high-performance similarity search
- **Embedding Processing**: Batch processing and normalization of music descriptions into searchable vectors
- **REST API**: Flask-based API for programmatic access to recommendations
- **Memory Optimization**: Designed to run efficiently on 512MB RAM deployments

### Core Features

- **Natural Language Search**: Query with phrases like "upbeat rock music" or "melancholic indie songs"
- **Vector Similarity Search**: Fast cosine similarity search using pgvector extension
- **Auto-Initialization**: Database schema and sample data setup automatically on first run
- **Music Service Integration**: Direct links to YouTube and Spotify for each recommendation

### Code Architecture

#### 1. Configuration Management (`src/config.py`)
Centralizes all application settings with environment variable support:

```python
class Config:
    # Database settings
    DATABASE_URL = os.getenv('DATABASE_URL')
    DB_MIN_POOL_SIZE = int(os.getenv('DB_MIN_POOL_SIZE', '1'))
    
    # Model configuration
    SENTENCE_TRANSFORMER_MODEL = os.getenv('SENTENCE_TRANSFORMER_MODEL', 'paraphrase-MiniLM-L3-v2')
    
    # Memory optimizations
    OPTIMIZE_FOR_MEMORY = os.getenv('OPTIMIZE_FOR_MEMORY', 'false').lower() == 'true'
```

**Key Features:**
- Environment-based configuration
- Memory optimization settings
- Deployment environment detection
- Validation and error handling

#### 2. Recommendation Engine (`src/recommendation_engine.py`)
Core semantic search functionality using pgvector:

```python
class MusicRecommendationEngine:
    async def get_recommendations(self, query: str, limit: int = 5) -> List[Dict]:
        """Get music recommendations based on natural language query."""
        
        # Convert query to normalized embedding vector
        query_embedding = self.model.encode(query)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Perform semantic similarity search using pgvector
        search_query = """
        SELECT song_id, song_name, band, description,
               (embedding <-> $1::vector) as raw_distance,
               ROUND(CAST((1.0 - (embedding <-> $1::vector) / 2.0) * 100.0 AS numeric), 1) as similarity_score
        FROM songs 
        ORDER BY embedding <-> $1::vector ASC
        LIMIT $2
        """
```

**Key Features:**
- **Semantic Search**: Converts natural language queries to vector embeddings
- **pgvector Integration**: Uses PostgreSQL's vector extension for fast similarity search
- **Dual API Support**: Both async and sync methods for different deployment needs
- **Memory Optimization**: Includes garbage collection and memory management
- **Error Handling**: Comprehensive error handling for production use

### Application Flow

1. **Initialization** (`app.py`):
   - Loads configuration from environment variables
   - Initializes sentence transformer model
   - Creates database connection pool
   - Sets up database schema and sample data

2. **Query Processing**:
   - User submits natural language query via API
   - Query is converted to normalized vector embedding
   - pgvector performs cosine similarity search
   - Results are formatted with music service links

3. **Response Generation**:
   - Similarity scores are calculated and normalized
   - Music service URLs are generated
   - JSON response is returned with recommendations

### Memory Optimization

The application is optimized to run on 512MB RAM deployments:

#### Model Selection
- Uses `paraphrase-MiniLM-L3-v2` (30MB model vs 400MB+ alternatives)
- CPU-only PyTorch installation to save memory
- Model caching in `/tmp` directory

#### Database Optimization
- Single database connection pool (vs multiple connections)
- Reduced batch sizes for processing
- Optimized query limits

#### Runtime Optimization
```python
# Environment variables for memory optimization
TOKENIZERS_PARALLELISM=false
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
PYTHONDONTWRITEBYTECODE=1
```

### Render.yaml Configuration

The project includes a sophisticated `render.yaml` file that's specifically optimized for Render's 512MB starter plan. This configuration demonstrates advanced deployment patterns for memory-constrained environments.

#### Key Features of render.yaml

**Build Process**
```yaml
buildCommand: |
  # Install only essential packages, no pandas!
  pip install --no-cache-dir Flask==3.0.0 gunicorn==21.2.0
  pip install --no-cache-dir torch>=2.1.2 --index-url https://download.pytorch.org/whl/cpu
  # Download model during build to avoid startup delays
  python3 -c "
  from sentence_transformers import SentenceTransformer
  model = SentenceTransformer('paraphrase-MiniLM-L3-v2', cache_folder='/tmp/sentence_transformers', device='cpu')
  "
```

**Memory-Optimized Runtime**
```yaml
startCommand: |
  # Set memory-optimized Python flags
  export PYTHONUNBUFFERED=1
  export TOKENIZERS_PARALLELISM=false
  export OMP_NUM_THREADS=1
  
  # Start with minimal workers and aggressive memory limits
  exec gunicorn --bind 0.0.0.0:$PORT app:app \
    --workers 1 \
    --max-requests 20 \
    --worker-tmp-dir /dev/shm
```

**Critical Environment Variables**
- `SENTENCE_TRANSFORMER_MODEL=paraphrase-MiniLM-L3-v2` - Ultra-small 30MB model
- `OPTIMIZE_FOR_MEMORY=true` - Enables aggressive memory management
- `DB_MAX_POOL_SIZE=1` - Single database connection to save memory
- `TOKENIZERS_PARALLELISM=false` - Disables parallelism to reduce memory usage

#### Database Configuration
The render.yaml automatically provisions:
- PostgreSQL 15 with pgvector extension
- Free tier database with automatic connection string injection
- Proper database naming and user configuration

#### Why This Configuration Works
1. **Build-Time Model Caching**: Downloads the ML model during build, not runtime
2. **CPU-Only PyTorch**: Saves ~200MB by avoiding CUDA dependencies
3. **Minimal Dependencies**: Excludes pandas and other heavy libraries
4. **Single Worker Process**: Reduces memory overhead
5. **Shared Memory**: Uses `/dev/shm` for temporary files

This render.yaml demonstrates how to deploy sophisticated ML applications on budget-friendly hosting plans.

### Troubleshooting

#### Database Connection Errors
```bash
# Check if pgvector extension is installed
psql $DATABASE_URL -c "SELECT * FROM pg_extension WHERE extname = 'vector';"

# Install pgvector if missing
psql $DATABASE_URL -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

#### Memory Issues on Render
- Ensure `OPTIMIZE_FOR_MEMORY=true` is set
- Use the smallest model: `paraphrase-MiniLM-L3-v2`
- Monitor memory usage in Render logs

#### Model Loading Failures
```bash
# Clear model cache
rm -rf /tmp/sentence_transformers/*

# Verify model download
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-MiniLM-L3-v2')"
```

#### Low Similarity Scores
- Try different query phrasings
- Check if sample data matches your query domain
- Consider the model limitations with music-specific terms