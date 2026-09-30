FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements (CPU torch by default)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application repository
COPY . .

# Ensure import resolution for ariadne modules
ENV PYTHONPATH=/app

# Expose port for demo application
EXPOSE 8501

# Launch Streamlit demo
CMD ["streamlit", "run", "versioning/demo/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
