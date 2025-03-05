FROM python:3.11.10
 
WORKDIR /app
 
# Install dependencies, including PortAudio and necessary build tools
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    python-pyaudio \
    libasound2-dev \
    libjack-dev \
    gcc \
    python3-dev \
&& rm -rf /var/lib/apt/lists/*
 
# Set environment variables to avoid audio errors in headless environments
ENV PYAUDIO_HOME=/usr
ENV SDL_AUDIODRIVER=dummy
 
# Copy and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
 
# Copy application code
COPY app.py .
 
# Expose Streamlit's default port
EXPOSE 8501
 
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]