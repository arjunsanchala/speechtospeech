FROM python:3.11.10

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    libasound2-dev \
    alsa-utils \
    libjack-dev \
    gcc \
    python3-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables to avoid audio device issues
ENV PYAUDIO_HOME=/usr
ENV SDL_AUDIODRIVER=dummy

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --only-binary :all: pyaudio && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .

# Expose Streamlit's default port
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
