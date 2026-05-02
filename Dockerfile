FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies (required for OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt requirements_web.txt /app/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -r requirements_web.txt

# Copy project
COPY . /app/

# Expose Streamlit port
EXPOSE 8501

# Command to run Streamlit app
CMD ["streamlit", "run", "web_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
