# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies in stages to handle PyTorch CPU-only properly
RUN pip install --no-cache-dir Flask==2.3.3 Flask-CORS==4.0.0 gunicorn==21.2.0
RUN pip install --no-cache-dir asyncpg==0.29.0 psycopg2-binary==2.9.7
RUN pip install --no-cache-dir numpy==1.24.3 pandas==2.0.3
RUN pip install --no-cache-dir kaggle==1.5.16 python-dotenv==1.0.0

# Install PyTorch CPU-only version and sentence-transformers with compatible versions
RUN pip install --no-cache-dir torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir sentence-transformers==2.2.2

# Skip model pre-download to avoid version conflicts - use lazy loading instead

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Expose port
EXPOSE 10000

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=10000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Start command
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app", "--workers", "1", "--timeout", "120", "--worker-class", "sync"]
