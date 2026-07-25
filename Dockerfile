FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies (required for OpenCV and healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt requirements_web.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt -r requirements_web.txt

# Copy project
COPY . /app/

# Expose Streamlit port
EXPOSE 8501

# Add Docker Healthcheck for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=20s \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Command to run Streamlit app
CMD ["streamlit", "run", "web_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

