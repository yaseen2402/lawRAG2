# base_images/python_ml_base/Dockerfile
FROM python:3.9-slim

# Set the working directory in the container (optional for base, but consistent)
WORKDIR /app

# Copy shared requirements (assuming requirements.txt contains sentence-transformers)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Download the model in the base image ---
ENV TORCH_HOME=/root/.cache/torch
RUN mkdir -p ${TORCH_HOME}/sentence_transformers
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"
# --- Model is now baked into this base image ---

# You might add other common tools or dependencies here