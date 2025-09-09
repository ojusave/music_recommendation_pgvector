"""
Music Recommendation Web App - Semantic search using pgvector with PostgreSQL.

This app demonstrates how AI can understand natural language queries like "sad piano music"
and find similar songs using vector similarity search in PostgreSQL.

Key concepts:
- Vectors: Lists of numbers that represent text meaning
- pgvector: PostgreSQL extension for vector similarity search
- Semantic search: Finding similar meanings, not just exact text matches
"""

import asyncio, sys, logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify

#from flask_cors import CORS
from src import Config, MusicRecommendationEngine

# Set up logging to see what's happening
logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Create Flask web application
app = Flask(__name__)
app.config['SECRET_KEY'] = Config.SECRET_KEY
#CORS(app)  # Enable CORS for all routes

# Create the AI recommendation engine with pgvector optimizations
recommendation_engine = MusicRecommendationEngine(use_halfvec=Config.USE_HALFVEC)

@app.route('/')
def index():
    """Main page - serves the music recommendation interface."""
    return render_template('index.html')

@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    """
    API endpoint for music recommendations using semantic search.
    
    How it works:
    1. Takes your natural language query (like "sad piano music")
    2. Converts it to a vector (list of numbers representing meaning)
    3. Finds songs with similar vectors in the database
    4. Returns the most similar songs
    
    Input: {"query": "upbeat rock music", "limit": 5}
    Output: {"success": true, "recommendations": [...], "query": "...", "count": 5}
    """
    logger.info("=== API RECOMMEND CALLED ===")
    
    # Ensure the recommendation engine is initialized
    ensure_initialized()
    
    try:
        # Get the JSON data from the request
        data = request.get_json()
        logger.info(f"Request data: {data}")
        
        # Validate that we have a query
        if not data or 'query' not in data:
            logger.error("Missing query parameter")
            return jsonify({'success': False, 'error': 'Missing query parameter - send {"query": "your search"}'}), 400
        
        query = data['query'].strip()
        logger.info(f"Processing search query: '{query}'")
        if not query:
            logger.error("Empty query")
            return jsonify({'success': False, 'error': 'Query cannot be empty - try "happy songs" or "rock music"'}), 400
        
        # Set limit with bounds checking (how many results to return)
        limit = data.get('limit', Config.DEFAULT_RECOMMENDATION_LIMIT)
        if not isinstance(limit, int) or limit < 1 or limit > Config.MAX_RECOMMENDATION_LIMIT:
            limit = Config.DEFAULT_RECOMMENDATION_LIMIT
        
        # Get recommendations using semantic search (this is where the AI magic happens!)
        logger.info(f"Calling get_recommendations_sync with query='{query}', limit={limit}")
        recommendations = recommendation_engine.get_recommendations_sync(query, limit)
        logger.info(f"Successfully got {len(recommendations)} recommendations")
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'query': query,
            'count': len(recommendations)
        })
        
    except MemoryError as e:
        logger.error(f"Memory error in recommendation API: {e}")
        return jsonify({'success': False, 'error': 'Server overloaded - try setting OPTIMIZE_FOR_MEMORY=true'}), 503
    except RuntimeError as e:
        error_msg = str(e)
        if "DATABASE_URL" in error_msg:
            return jsonify({'success': False, 'error': 'Database connection failed. Check your DATABASE_URL environment variable.'}), 500
        elif "pgvector" in error_msg:
            return jsonify({'success': False, 'error': 'pgvector extension missing. Run: CREATE EXTENSION IF NOT EXISTS vector;'}), 500
        else:
            return jsonify({'success': False, 'error': error_msg}), 500
    except Exception as e:
        logger.error(f"Recommendation API error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': 'Internal server error - check logs for details'}), 500

@app.route('/api/status')
def api_status():
    """Status endpoint for monitoring database health and statistics."""
    try:
        # Ensure the recommendation engine is initialized
        ensure_initialized()
        
        stats = recommendation_engine.get_database_stats_sync()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Status API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Simple health check endpoint for deployment monitoring."""
    return jsonify({
        'status': 'healthy', 
        'service': 'music-recommendations',
        'message': 'App is running'
    })

@app.route('/api/test', methods=['GET', 'POST'])
def api_test():
    """Test endpoint to check if API is reachable."""
    logger.info(f"=== API TEST CALLED - Method: {request.method} ===")
    return jsonify({
        'status': 'API is working',
        'method': request.method,
        'timestamp': str(datetime.now())
    })

def initialize_app():
    """Initialize the application components with async compatibility."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(recommendation_engine.initialize())
    except RuntimeError:
        asyncio.run(recommendation_engine.initialize())

# Initialize app when module is loaded (except for testing)
if __name__ != '__main__' and 'test' not in sys.modules:
    initialize_app()

# For Flask development server, ensure initialization happens synchronously
def ensure_initialized():
    """Ensure the recommendation engine is initialized before handling requests."""
    if not hasattr(recommendation_engine, 'connection_pool') or recommendation_engine.connection_pool is None:
        logger.info("Initializing recommendation engine...")
        asyncio.run(recommendation_engine.initialize())
        logger.info("Recommendation engine ready!")

if __name__ == '__main__':
    """Main entry point for local development."""
    initialize_app()
    app.run(debug=Config.DEBUG, host=Config.HOST, port=Config.PORT)