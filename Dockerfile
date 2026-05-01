# OmnissiahCore — Dockerfile
# Lean runtime image. No torch, no CUDA, no build bloat.
# Large files (Db/, bge_m3_onnx/) are pulled from Hugging Face at startup.
#
# Build:  docker build -t omnissiahcore .
# Run:    docker run -p 8000:8000 -e HF_TOKEN=your_token omnissiahcore

FROM python:3.10-slim

# System deps — only what FAISS and sentence-transformers actually need
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (separate layer — only rebuilds if requirements change)
COPY requirements.docker.txt .
RUN pip install --no-cache-dir -r requirements.docker.txt

# Copy only runtime code — nothing else
COPY Api/         Api/
COPY Core/        Core/
COPY Scripts/     Scripts/
COPY app_text.json config.json main.py ./

# Empty dirs — filled at runtime by pull_index.py
RUN mkdir -p Db bge_m3_onnx

EXPOSE 8000

# On start: pull missing large files from HF, then launch API
CMD ["sh", "-c", "python Scripts/pull_index.py && python main.py api"]