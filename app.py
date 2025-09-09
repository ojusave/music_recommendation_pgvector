"""
Music Recommendation Web App - Semantic search using pgvector with PostgreSQL.
Demonstrates production-ready vector similarity search with natural language queries.
"""

import asyncio, sys, logging
from flask import Flask, render_template, request, jsonify

from flask_cors import CORS
from src import Config, MusicRecommendationEngine

logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = Config.SECRET_KEY
CORS(app)  # Enable CORS for all routes
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
    try:
        data = request.get_json()
        
        # Validate input
        if not data or 'query' not in data:
            return jsonify({'success': False, 'error': 'Missing query parameter'}), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({'success': False, 'error': 'Query cannot be empty'}), 400
        
        # Set limit with bounds checking
        limit = data.get('limit', Config.DEFAULT_RECOMMENDATION_LIMIT)
        if not isinstance(limit, int) or limit < 1 or limit > Config.MAX_RECOMMENDATION_LIMIT:
            limit = Config.DEFAULT_RECOMMENDATION_LIMIT
        
        # Get recommendations using semantic search with timeout
        try:
            recommendations = asyncio.wait_for(
                recommendation_engine.get_recommendations(query, limit),
                timeout=25.0  # 25 second timeout
            )
            recommendations = asyncio.run(recommendations)
        except asyncio.TimeoutError:
            logger.error(f"Search timeout for query: {query}")
            return jsonify({'success': False, 'error': 'Search timeout - try a simpler query'}), 408
        
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
        stats = asyncio.run(recommendation_engine.get_database_stats())
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Status API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Simple health check endpoint for deployment monitoring."""
    try:
        stats = asyncio.run(recommendation_engine.get_database_stats())
        if 'error' in stats:
            return jsonify({
                'status': 'healthy', 
                'service': 'music-recommendations',
                'database': 'not_initialized',
                'message': 'App is running but database needs setup'
            })
        return jsonify({
            'status': 'healthy', 
            'service': 'music-recommendations',
            'database': 'ready',
            'songs': stats.get('total_songs', 0)
        })
    except Exception:
        return jsonify({
            'status': 'healthy', 
            'service': 'music-recommendations',
            'database': 'unknown',
            'message': 'App is running but database status unclear'
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

if __name__ == '__main__':
    """Main entry point for local development."""
    initialize_app()
    app.run(debug=Config.DEBUG, host=Config.HOST, port=Config.PORT)