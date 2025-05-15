# Law RAG System with ChromaDB

This project demonstrates a Retrieval Augmented Generation (RAG) system designed for querying legal cases. It utilizes a multi-vector retrieval strategy with ChromaDB as the vector store, Sentence Transformers for embeddings, and Google Gemini as the Large Language Model. The system is containerized using Docker and managed with Docker Compose.

The project consists of two main independent components:
1.  **Indexer:** Populates the ChromaDB vector store with legal case data.
2.  **Retriever:** Performs RAG queries against the populated ChromaDB instance using a custom retrieval strategy.


## Project Structure
```
lawRAG/

├── base_images/
│   └── rag_base
│       └── Dockerfile
│       └── requirements.txt
├── data/
│   └── cases_1.jsonl           #legal case data file (JSONL format)
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
```

## Running with Docker Compose

Navigate to the `lawRAG/` directory in your terminal.

1.  **Build the Docker Images:**
    - docker-compose build base_images
    
    - docker-compose build

2.  **Run the Indexer (One-off Job):**
    
    docker-compose run indexer
    
    Wait for this process to complete before running the retriever. You should see output indicating the data loading and indexing progress.

3.  **Run the Retriever (Interactive Q&A):**
    This command starts a container based on the retriever image. It will connect to the ChromaDB instance (using the data persisted in `lawRAG/chroma_db_storage` on your host) and starts the gradio instance which can be accessed in http://localhost:7860

    docker-compose run -p 7860:7860 retriever





1.  **Build the Docker Images:**
    * Build the base image (ensure your `requirements.txt` includes `qdrant-client` and is up-to-date):
        ```bash
        docker-compose build base_images
        ```
    * Build the images for the Qdrant indexer and retriever services:
        ```bash
        docker-compose build indexer_qdrant retriever_qdrant
        ```
        *(Alternatively, `docker-compose build` will build all services defined in the YAML.)*

2.  **Run the Indexer (One-off Job):**
    This command starts the `indexer_qdrant` container. It will automatically start the `qdrant` database service if it's not already running due to the `depends_on` configuration. The indexer connects to the Qdrant service and populates the database.
    ```bash
    docker-compose run indexer_qdrant
    ```
    *Adding `--rm` is good practice for one-off jobs like indexing; it automatically removes the container after it finishes.*

    Wait for this process to complete before running the retriever. You should see output indicating the data loading and indexing progress directly in your terminal.

3.  **Run the Retriever (Interactive Q&A):**
    This command starts the `retriever_qdrant` container. It will also automatically start the `qdrant` database service if needed. The retriever connects to the Qdrant service to fetch data for answers and starts the Gradio instance.
    ```bash
    docker-compose up retriever_qdrant
    ```
    *(Using `docker-compose up` is standard for starting services meant to run continuously. It also handles the `depends_on` on the `qdrant` service.)*

    The Gradio interface will be accessible on your host machine at `http://localhost:7860`.

    To run the retriever and Qdrant database in the background (detached mode), use:
    ```bash
    docker-compose up -d retriever_qdrant
    ```

    To stop the running services (Qdrant and Retriever) when running in detached mode, use:
    ```bash
    docker-compose down retriever_qdrant
    ```
    (or `docker-compose down` from the `lawRAG/` directory to stop all services).

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

