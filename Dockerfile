FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements (CPU torch by default)
COPY ariadne/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application repository
COPY . /app

# Ensure import resolution for ariadne modules
ENV PYTHONPATH=/app

# Expose port for Streamlit demo application
EXPOSE 8501

# Healthcheck to verify service availability
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit demo
CMD ["streamlit", "run", "ariadne/versioning/demo/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
