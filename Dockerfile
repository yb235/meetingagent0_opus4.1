FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.7.0

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root --no-dev

# Copy application code
COPY api/ ./api/
COPY config/ ./config/
COPY .env.example ./.env

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "api.main"]
