/**
 * Music Recommendation App - Frontend JavaScript
 * ==============================================
 * 
 * This file handles the frontend interactions for the music recommendation app.
 * It demonstrates best practices for:
 * - Making async API calls to Flask backend
 * - Handling loading states and user feedback
 * - Dynamic content rendering
 * - Error handling and user experience
 * 
 * Key features:
 * - Real-time search as user types
 * - Example query suggestions
 * - Loading indicators
 * - Responsive results display
 * - Error handling with retry functionality
 */

class MusicRecommendationApp {
    constructor() {
        this.searchInput = document.getElementById('searchInput');
        this.searchButton = document.getElementById('searchButton');
        this.limitSelect = document.getElementById('limitSelect');
        this.resultsSection = document.getElementById('resultsSection');
        this.resultsContainer = document.getElementById('resultsContainer');
        this.errorSection = document.getElementById('errorSection');
        this.errorText = document.getElementById('errorText');
        this.retryButton = document.getElementById('retryButton');
        
        this.isLoading = false;
        this.currentQuery = '';
        
        this.initializeEventListeners();
        this.initializeExampleQueries();
    }
    
    /**
     * Initialize all event listeners for user interactions
     */
    initializeEventListeners() {
        // Search button click
        this.searchButton.addEventListener('click', () => {
            this.performSearch();
        });
        
        // Enter key in search input
        this.searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.performSearch();
            }
        });
        
        // Input validation and character counter
        this.searchInput.addEventListener('input', (e) => {
            const value = e.target.value;
            if (value.length > 200) {
                e.target.value = value.substring(0, 200);
            }
        });
        
        // Retry button
        this.retryButton.addEventListener('click', () => {
            this.hideError();
            this.performSearch();
        });
        
        // Auto-focus search input on page load
        window.addEventListener('load', () => {
            this.searchInput.focus();
        });
    }
    
    /**
     * Initialize example query buttons with click handlers
     */
    initializeExampleQueries() {
        const exampleButtons = document.querySelectorAll('.example-query');
        exampleButtons.forEach(button => {
            button.addEventListener('click', () => {
                const query = button.getAttribute('data-query');
                this.searchInput.value = query;
                this.performSearch();
            });
        });
    }
    
    /**
     * Perform a music recommendation search
     */
    async performSearch() {
        const query = this.searchInput.value.trim();
        
        // Validation
        if (!query) {
            this.showError('Please enter a search query');
            return;
        }
        
        if (query.length < 3) {
            this.showError('Search query must be at least 3 characters long');
            return;
        }
        
        // Prevent multiple simultaneous searches
        if (this.isLoading) {
            return;
        }
        
        this.currentQuery = query;
        this.setLoadingState(true);
        this.hideError();
        this.hideResults();
        
        try {
            const limit = parseInt(this.limitSelect.value);
            const recommendations = await this.fetchRecommendations(query, limit);
            
            if (recommendations && recommendations.length > 0) {
                this.displayResults(recommendations, query);
            } else {
                this.showError('No recommendations found. Try a different search query.');
            }
            
        } catch (error) {
            console.error('Search error:', error);
            this.showError(this.getErrorMessage(error));
        } finally {
            this.setLoadingState(false);
        }
    }
    
    /**
     * Fetch recommendations from the backend API
     * 
     * @param {string} query - User's search query
     * @param {number} limit - Number of recommendations to fetch
     * @returns {Promise<Array>} - Array of recommendation objects
     */
    async fetchRecommendations(query, limit) {
        const response = await fetch('/api/recommend', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                limit: limit
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.error || 'Unknown error occurred');
        }
        
        return data.recommendations;
    }
    
    /**
     * Display search results in the UI
     * 
     * @param {Array} recommendations - Array of recommendation objects
     * @param {string} query - Original search query
     */
    displayResults(recommendations, query) {
        // Clear previous results
        this.resultsContainer.innerHTML = '';
        
        // Create result cards
        recommendations.forEach((rec, index) => {
            const resultCard = this.createResultCard(rec, index + 1);
            this.resultsContainer.appendChild(resultCard);
        });
        
        // Show results section with smooth animation
        this.resultsSection.style.display = 'block';
        setTimeout(() => {
            this.resultsSection.scrollIntoView({ 
                behavior: 'smooth', 
                block: 'start' 
            });
        }, 100);
    }
    
    /**
     * Create a result card element for a single recommendation
     * 
     * @param {Object} recommendation - Recommendation data
     * @param {number} rank - Ranking position (1-based)
     * @returns {HTMLElement} - Result card element
     */
    createResultCard(recommendation, rank) {
        const card = document.createElement('div');
        card.className = 'result-card';
        
        // Sanitize text content to prevent XSS
        const sanitize = (text) => {
            const div = document.createElement('div');
            div.textContent = text || '';
            return div.innerHTML;
        };
        
        // Format similarity score with color coding
        const similarity = recommendation.similarity_score;
        let scoreClass = '';
        if (similarity >= 80) scoreClass = 'high-score';
        else if (similarity >= 60) scoreClass = 'medium-score';
        else scoreClass = 'low-score';
        
        card.innerHTML = `
            <div class="result-header">
                <div class="result-rank">${rank}</div>
                <div class="result-info">
                    <h3 class="result-title">${sanitize(recommendation.song_name)}</h3>
                    <p class="result-artist">by ${sanitize(recommendation.artist)}</p>
                </div>
            </div>
            
            ${recommendation.description ? 
                `<p class="result-description">${sanitize(recommendation.description)}</p>` : 
                ''
            }
            
            <div class="result-score ${scoreClass}">
                <span>Match: ${similarity}%</span>
                <span class="result-distance">Distance: ${recommendation.raw_distance}</span>
            </div>
            
            <div class="result-links">
                <a href="${recommendation.youtube_url}" 
                   target="_blank" 
                   rel="noopener noreferrer" 
                   class="music-link youtube">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M23.498 6.186a2.999 2.999 0 0 0-2.112-2.112C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.386.529A2.999 2.999 0 0 0 .502 6.186C0 8.067 0 12 0 12s0 3.933.502 5.814a2.999 2.999 0 0 0 2.112 2.112C4.495 20.455 12 20.455 12 20.455s7.505 0 9.386-.529a2.999 2.999 0 0 0 2.112-2.112C24 15.933 24 12 24 12s0-3.933-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/>
                    </svg>
                    YouTube
                </a>
                <a href="${recommendation.spotify_url}" 
                   target="_blank" 
                   rel="noopener noreferrer" 
                   class="music-link spotify">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.84-.179-.959-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.361 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.42 1.56-.299.421-1.02.599-1.559.3z"/>
                    </svg>
                    Spotify
                </a>
            </div>
        `;
        
        // Add click tracking for analytics (if needed)
        card.addEventListener('click', () => {
            this.trackResultClick(recommendation, rank);
        });
        
        return card;
    }
    
    /**
     * Set loading state for the search button and UI
     * 
     * @param {boolean} loading - Whether the app is in loading state
     */
    setLoadingState(loading) {
        this.isLoading = loading;
        
        const buttonText = this.searchButton.querySelector('.button-text');
        const spinner = this.searchButton.querySelector('.loading-spinner');
        
        if (loading) {
            buttonText.style.display = 'none';
            spinner.style.display = 'block';
            this.searchButton.disabled = true;
            this.searchInput.disabled = true;
        } else {
            buttonText.style.display = 'block';
            spinner.style.display = 'none';
            this.searchButton.disabled = false;
            this.searchInput.disabled = false;
        }
    }
    
    /**
     * Show error message to user
     * 
     * @param {string} message - Error message to display
     */
    showError(message) {
        this.errorText.textContent = message;
        this.errorSection.style.display = 'block';
        this.hideResults();
        
        // Scroll to error message
        setTimeout(() => {
            this.errorSection.scrollIntoView({ 
                behavior: 'smooth', 
                block: 'center' 
            });
        }, 100);
    }
    
    /**
     * Hide error message
     */
    hideError() {
        this.errorSection.style.display = 'none';
    }
    
    /**
     * Hide results section
     */
    hideResults() {
        this.resultsSection.style.display = 'none';
    }
    
    /**
     * Get user-friendly error message from error object
     * 
     * @param {Error} error - Error object
     * @returns {string} - User-friendly error message
     */
    getErrorMessage(error) {
        if (error.message.includes('Failed to fetch')) {
            return 'Unable to connect to the server. Please check your internet connection and try again.';
        }
        
        if (error.message.includes('500')) {
            return 'Server error occurred. Please try again in a moment.';
        }
        
        if (error.message.includes('404')) {
            return 'Service not found. Please refresh the page and try again.';
        }
        
        return error.message || 'An unexpected error occurred. Please try again.';
    }
    
    /**
     * Track result clicks for analytics (placeholder)
     * 
     * @param {Object} recommendation - Clicked recommendation
     * @param {number} rank - Ranking position
     */
    trackResultClick(recommendation, rank) {
        // Placeholder for analytics tracking
        // In a real app, you might send this to Google Analytics, Mixpanel, etc.
        console.log('Result clicked:', {
            query: this.currentQuery,
            song: recommendation.song_name,
            artist: recommendation.artist,
            rank: rank,
            similarity: recommendation.similarity_score
        });
    }
}

/**
 * Utility functions for the app
 */
const Utils = {
    /**
     * Debounce function to limit API calls
     * 
     * @param {Function} func - Function to debounce
     * @param {number} wait - Wait time in milliseconds
     * @returns {Function} - Debounced function
     */
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },
    
    /**
     * Format number with commas for better readability
     * 
     * @param {number} num - Number to format
     * @returns {string} - Formatted number string
     */
    formatNumber(num) {
        return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    }
};

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const app = new MusicRecommendationApp();
    
    // Add global error handler
    window.addEventListener('error', (event) => {
        console.error('Global error:', event.error);
    });
    
    // Add unhandled promise rejection handler
    window.addEventListener('unhandledrejection', (event) => {
        console.error('Unhandled promise rejection:', event.reason);
    });
    
    // Expose app to global scope for debugging
    window.musicApp = app;
});

// Service Worker registration for PWA capabilities (optional)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then((registration) => {
                console.log('SW registered: ', registration);
            })
            .catch((registrationError) => {
                console.log('SW registration failed: ', registrationError);
            });
    });
}
