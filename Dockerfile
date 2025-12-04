# AI-based Intrusion Detection System Dockerfile

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpcap-dev \
    nmap \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p models

# Expose port for Streamlit
EXPOSE 8501

# Command to run the dashboard
CMD ["streamlit", "run", "ids_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
