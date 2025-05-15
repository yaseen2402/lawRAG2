# Use a slim Python base image
FROM rag_base

# Set the working directory in the container
WORKDIR /app

# Create directory for mounted ChromaDB data inside the container
# /app/chroma_db_storage is where the ChromaDB data will be mounted
RUN mkdir /app/chroma_db_storage


# Install evaluation dependencies
# Assuming ragas and datasets are needed based on your eval.py imports
RUN pip install ragas datasets gradio bert-score

# Copy the evaluation script and data into the working directory
COPY ./eval/eval.py .
COPY ./eval/evaluation_data.json . 
COPY ./retriever/query.py . 
# The command to run the script when the container starts
CMD ["python", "eval.py"]