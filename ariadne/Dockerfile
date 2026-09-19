FROM python:3.12-slim

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

# Expose port for demo application (Person D)
EXPOSE 7860

# Default command: show setup status
CMD ["python", "-c", "import yaml; print('Ariadne environment ready.'); print(yaml.safe_load(open('config.yaml'))['global'])"]
