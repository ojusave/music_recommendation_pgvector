"""
Music Recommendation Web App
============================

A Flask web application that demonstrates semantic music search using pgvector
with PostgreSQL. This app demonstrates how to build
production-ready vector similarity search applications.

Key Features:
- Natural language music queries
- Semantic similarity search with pgvector
- Automatic database setup and sample data loading
- YouTube and Spotify integration
- Clean, lightweight frontend

Built for deployment with PostgreSQL + pgvector extension.
"""

import asyncio
import sys
from flask import Flask, render_template, request, jsonify
import logging

# Import our modular components
from src import Config, MusicRecommendationEngine

# Configure logging
logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = Config.SECRET_KEY

# Initialize the recommendation engine
recommendation_engine = MusicRecommendationEngine()

@app.route('/')
def index():
    """
    Main page - serves the music recommendation interface.
    
    This demonstrates a simple but effective UI for semantic search applications.
    """
    return render_template('index.html')

@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    """
    API endpoint for getting music recommendations.
    
    This endpoint demonstrates:
    1. Input validation for user queries
    2. Async database operations with proper error handling
    3. JSON API responses with structured data
    4. Performance logging for monitoring
    
    Expected JSON input:
    {
        "query": "upbeat rock music for working out",
        "limit": 5
    }
    
    Returns JSON response:
    {
        "success": true,
        "recommendations": [...],
        "query": "original query",
        "count": 5
    }
    """
    try:
        data = request.get_json()
        
        # Validate required parameters
        if not data or 'query' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing query parameter'
            }), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({
                'success': False,
                'error': 'Query cannot be empty'
            }), 400
        
        # Validate and set limit parameter
        limit = data.get('limit', Config.DEFAULT_RECOMMENDATION_LIMIT)
        if not isinstance(limit, int) or limit < 1 or limit > Config.MAX_RECOMMENDATION_LIMIT:
            limit = Config.DEFAULT_RECOMMENDATION_LIMIT
        
        # Get recommendations using our semantic search engine
        recommendations = asyncio.run(recommendation_engine.get_recommendations(query, limit))
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'query': query,
            'count': len(recommendations)
        })
        
    except Exception as e:
        logger.error(f"Recommendation API error: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@app.route('/api/status')
def api_status():
    """
    Status endpoint for monitoring database health and statistics.
    
    Useful for:
    - Monitoring database connectivity
    - Checking data availability
    - Debugging deployment issues
    """
    try:
        stats = asyncio.run(recommendation_engine.get_database_stats())
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Status API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """
    Simple health check endpoint for deployment monitoring.
    
    This endpoint provides information about:
    - Service health status
    - Database connectivity
    - Data availability
    """
    try:
        # Check if database is initialized
        stats = asyncio.run(recommendation_engine.get_database_stats())
        if 'error' in stats:
            return jsonify({
                'status': 'healthy', 
                'service': 'music-recommendations',
                'database': 'not_initialized',
                'message': 'App is running but database needs setup'
            })
        else:
            return jsonify({
                'status': 'healthy', 
                'service': 'music-recommendations',
                'database': 'ready',
                'songs': stats.get('total_songs', 0)
            })
    except Exception as e:
        return jsonify({
            'status': 'healthy', 
            'service': 'music-recommendations',
            'database': 'unknown',
            'message': 'App is running but database status unclear'
        })

def initialize_app():
    """
    Initialize the application components.
    
    This function handles the async initialization of the recommendation engine
    in a way that's compatible with Flask's synchronous nature.
    """
    try:
        # Check if we're already in an event loop
        loop = asyncio.get_running_loop()
        # If we are, create a task instead
        loop.create_task(recommendation_engine.initialize())
    except RuntimeError:
        # No event loop running, safe to use asyncio.run()
        asyncio.run(recommendation_engine.initialize())

# Initialize the app when module is loaded
# But only if we're not being imported for testing
if __name__ != '__main__' and 'test' not in sys.modules:
    initialize_app()

if __name__ == '__main__':
    """
    Main entry point for local development.
    
    For production deployment, this section won't be used
    as production servers typically use gunicorn to serve the application.
    """
    # Initialize the recommendation engine
    initialize_app()
    
    # Start the Flask development server
    app.run(
        debug=Config.DEBUG, 
        host=Config.HOST, 
        port=Config.PORT
    )