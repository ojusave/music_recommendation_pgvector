"""
Music Recommendation Web App - Semantic search using pgvector with PostgreSQL.
Demonstrates production-ready vector similarity search with natural language queries.
"""

import asyncio, sys, logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify

#from flask_cors import CORS
from src import Config, MusicRecommendationEngine

# Global recommendation engine (initialized on first use)
recommendation_engine = None

def get_recommendation_engine():
    """Get or initialize the recommendation engine."""
    global recommendation_engine
    if recommendation_engine is None:
        logger.info("Initializing recommendation engine...")
        recommendation_engine = MusicRecommendationEngine()
        logger.info("Recommendation engine ready")
    return recommendation_engine

logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = Config.SECRET_KEY
#CORS(app)  # Enable CORS for all routes
recommendation_engine = MusicRecommendationEngine()

@app.route('/')
def index():
    """Main page - serves the music recommendation interface."""
    return render_template('index.html')

@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    """
    API endpoint for music recommendations using semantic search.
    
    Input: {"query": "upbeat rock music", "limit": 5}
    Output: {"success": true, "recommendations": [...], "query": "...", "count": 5}
    """
    logger.info("=== API RECOMMEND CALLED ===")
    try:
        data = request.get_json()
        logger.info(f"Request data: {data}")
        
        # Validate input
        if not data or 'query' not in data:
            logger.error("Missing query parameter")
            return jsonify({'success': False, 'error': 'Missing query parameter'}), 400
        
        query = data['query'].strip()
        logger.info(f"Processing search query: '{query}'")
        if not query:
            logger.error("Empty query")
            return jsonify({'success': False, 'error': 'Query cannot be empty'}), 400
        
        # Set limit with bounds checking
        limit = data.get('limit', Config.DEFAULT_RECOMMENDATION_LIMIT)
        if not isinstance(limit, int) or limit < 1 or limit > Config.MAX_RECOMMENDATION_LIMIT:
            limit = Config.DEFAULT_RECOMMENDATION_LIMIT
        
        # Get recommendations using semantic search (synchronous)
        logger.info(f"Calling get_recommendations_sync with query='{query}', limit={limit}")
        engine = get_recommendation_engine()
        recommendations = engine.get_recommendations_sync(query, limit)
        logger.info(f"Successfully got {len(recommendations)} recommendations")
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'query': query,
            'count': len(recommendations)
        })
        
    except MemoryError as e:
        logger.error(f"Memory error in recommendation API: {e}")
        return jsonify({'success': False, 'error': 'Server overloaded - please try again'}), 503
    except Exception as e:
        logger.error(f"Recommendation API error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

@app.route('/api/status')
def api_status():
    """Status endpoint for monitoring database health and statistics."""
    try:
        engine = get_recommendation_engine()
        stats = engine.get_database_stats_sync()
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
    """Initialize the application components synchronously."""
    logger.info("Initializing recommendation engine...")
    # Skip async initialization since we're using sync methods now
    logger.info("Using synchronous database operations - no async initialization needed")

# Skip initialization during import to avoid worker timeout issues
# The recommendation engine will initialize on first use

if __name__ == '__main__':
    """Main entry point for local development."""
    initialize_app()
    app.run(debug=Config.DEBUG, host=Config.HOST, port=Config.PORT)