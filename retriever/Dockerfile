# Use a slim Python base image
FROM rag_base

# Set the working directory in the container
WORKDIR /app


# Create directory for mounted ChromaDB data inside the container
# /app/chroma_db_storage is where the ChromaDB data will be mounted
RUN mkdir /app/chroma_db_storage
RUN pip install gradio

# Copy the retriever script into the working directory
COPY ./retriever/eval_data.json .
COPY ./retriever/query.py . 
EXPOSE 7860

# The command to run the script when the container starts
CMD ["python", "query.py"]
