# Use a slim Python base image
FROM rag_base

# Set the working directory in the container
WORKDIR /app


# Create directories for mounted volumes inside the container
# /app/data is where the data file will be mounted
RUN mkdir /app/data
# /app/chroma_db_storage is where the ChromaDB data will be mounted
RUN mkdir /app/chroma_db_storage

# Copy the indexer script into the working directory
COPY ./indexer/index.py .

# The command to run the script when the container starts
CMD ["python", "index.py"]

