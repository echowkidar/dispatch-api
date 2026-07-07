FROM python:3.11-slim

WORKDIR /app

# Pillow needs a couple of system libs to decode jpeg/png properly
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py prompt.txt ./

EXPOSE 8000

# 1 worker is intentional: the whole point is to serialize load onto the
# single CPU-only Ollama instance behind it. The lock in main.py assumes
# a single process.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
