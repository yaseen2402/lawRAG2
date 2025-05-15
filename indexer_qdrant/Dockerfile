# indexer_qdrant/Dockerfile
# Use the base image that includes common dependencies (like qdrant-client)
FROM rag_base

# Set the working directory in the container
WORKDIR /app

# Create directory for mounted data volume
# /app/data is where the data file (cases_2.jsonl) will be mounted
RUN mkdir /app/data

RUN pip install --no-cache-dir qdrant-client


# Copy the Qdrant indexer script into the working directory
COPY ./indexer_qdrant.py .

# The command to run the Qdrant indexer script when the container starts
CMD ["python", "indexer_qdrant.py"]