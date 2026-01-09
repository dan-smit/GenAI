#~120 MB base image
FROM python:3.11-slim

#Sends output logs to terminal, 
ENV PYTHONUNBUFFERED=1 

WORKDIR /app

# Install minimal build tools 
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Use port 8501 for Streamlit
EXPOSE 8501

# Ensures Streamlit doesn't open a browser window in the container 
ENV STREAMLIT_SERVER_HEADLESS=true

CMD ["streamlit", "run", "stockanalyzer/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
