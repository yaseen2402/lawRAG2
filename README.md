# Law RAG System with ChromaDB

This project demonstrates a Retrieval Augmented Generation (RAG) system designed for querying legal cases. It utilizes a multi-vector retrieval strategy with ChromaDB as the vector store, Sentence Transformers for embeddings, and Google Gemini as the Large Language Model. The system is containerized using Docker and managed with Docker Compose.

The project consists of two main independent components:
1.  **Indexer:** Populates the ChromaDB vector store with legal case data.
2.  **Retriever:** Performs RAG queries against the populated ChromaDB instance using a custom retrieval strategy.


## Project Structure

lawRAG/
├── data/
│   └── cases_page_2.jsonl      #legal case data file (JSONL format)
├── indexer/
│   ├── index.py                # The script to build the ChromaDB index
│   └── Dockerfile              # Dockerfile for the indexer service
├── retriever/
│   ├── query.py                # The script for the RAG retriever and Q&A
│   └── Dockerfile              # Dockerfile for the retriever service
├── chroma_db_storage/          # This directory will be created by the indexer 
├── .env                        # Environment variables (API keys)
├── requirements.txt            # Shared Python dependencies for both services
└── docker-compose.yml          # Docker Compose file to manage services


## Running with Docker Compose

Navigate to the `lawRAG/` directory in your terminal.

1.  **Build the Docker Images:**
    This command builds the images for both the indexer and retriever services based on their respective Dockerfiles. You only need to do this once or when you change the Dockerfiles or `requirements.txt`.

    - docker-compose build base_images
    
    - docker-compose build

2.  **Run the Indexer (One-off Job):**
    This command starts a container based on the indexer image. It will run your `index.py`, read the data file, compute embeddings, and populate the ChromaDB database in the `lawRAG/chroma_db_storage` directory on your host machine via the volume mount. The container will automatically stop and be removed (`--rm`) after the script finishes.

    
    docker-compose run --rm indexer
    
    Wait for this process to complete before running the retriever. You should see output indicating the data loading and indexing progress.

3.  **Run the Retriever (Interactive Q&A):**
    This command starts a container based on the retriever image. It will connect to the ChromaDB instance (using the data persisted in `lawRAG/chroma_db_storage` on your host) and start the interactive Q&A loop.

    ```bash
    docker-compose up retriever
    ```
    The retriever script will start, and you can begin typing your legal questions into the terminal. Type `exit` to quit the retriever and stop the container (`Ctrl+C` also works).

    *Note: The retriever service's `restart: on-failure` policy in `docker-compose.yml` means Docker Compose will try to restart it if it crashes, but you stop it cleanly by typing `exit` in the application prompt.*

## Configuration

* **Environment Variables:** Set your `GOOGLE_API_KEY` in the `.env` file.
* **Retrieval Parameters:** You can adjust retrieval parameters like `INITIAL_SEARCH_K`, `MAX_FILTERED_K`, and `TOTAL_FILTER_QUERY_LIMIT` directly in the `retriever/query.py` file.
* **LLM Model:** The `llm_model_name` is configured in `query.py`.
* **ChromaDB Location:** The `persist_directory` in the Python scripts and the volume mount in `docker-compose.yml` (`./chroma_db_storage:/app/chroma_db_storage`) define where the database is stored on your host machine.

## Notes

* The embedding model used is `BAAI/bge-large-en-v1.5`. This model requires a deep learning backend like PyTorch (`torch`) or TensorFlow (`tensorflow`), specified in `requirements.txt`.
* The system implements a "Parent Document Retrieval" strategy conceptually, retrieving full case contexts based on initially relevant chunks to improve RAG quality.
* The interactive Q&A loop in the retriever will print the LLM's answer and then attempt to show the sources (the full case contexts retrieved).

---

