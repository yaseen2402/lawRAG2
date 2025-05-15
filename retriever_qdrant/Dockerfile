# retriever_qdrant/Dockerfile
# Use the base image that includes common dependencies
FROM rag_base

# Set the working directory in the container
WORKDIR /app

# Copy the Qdrant retriever script into the working directory
RUN pip install --no-cache-dir qdrant-client==1.14.2
RUN pip install gradio


COPY ./retriever_qdrant.py .

# The command to run the Qdrant retriever script when the container starts
CMD ["python", "-u", "retriever_qdrant.py"]
