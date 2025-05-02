import chromadb 
import json
from sentence_transformers import SentenceTransformer

from typing import List, Dict, Any, Union
import os
from dotenv import load_dotenv 
import time

load_dotenv() 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


file_path = r"/app/data/cases_2.jsonl" 

# --- ChromaDB Setup ---
collection_name = "law_rag" 
persist_directory = "/app/chroma_db_storage" 
os.makedirs(persist_directory, exist_ok=True)

# BAAI/bge-large-en-v1.5 outputs 1024 dimensions.
dimension = 1024
sentence_transformer_model_name = "BAAI/bge-large-en-v1.5"


# Use PersistentClient to save data to disk
print(f"Initializing ChromaDB client and persisting to: {persist_directory}")
client = chromadb.PersistentClient(path=persist_directory)

# Get or create the collection
try:
    print(f"Getting or creating ChromaDB collection: '{collection_name}'...")
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"} # Set the distance metric
    )
    print(f"Connected to collection '{collection.name}'. Total items: {collection.count()}")
except Exception as e:
    print(f"Error connecting to or creating ChromaDB collection: {e}")
    raise 

# Initialize the Sentence Transformer embedding model
print(f"Loading Sentence Transformer model: {sentence_transformer_model_name}...")
embedding_model = SentenceTransformer(sentence_transformer_model_name)
print("Embedding model loaded.")

def load_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f: 
            for line in f:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Skipping line due to JSON decode error: {e} - Line: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        return 

METADATA_FIELDS_TO_VECTORIZE = [
    "medium_neutral_citation",
    "catchwords",
    "category",
    "parties",
    "legislation_cited",
    "cases_cited",
    "jurisdiction",
]

def store_case_multi_vectors_in_chroma(file_path: str, collection: chromadb.Collection, embedding_model):
    """
    Loads case data, generates embeddings using SentenceTransformer,
    and stores them in a ChromaDB collection with associated metadata.
    Creates separate vectors for specified metadata fields and paragraphs.
    """
    print(f"Loading data from {file_path}...")
    data = load_data(file_path)

    if data is None:
        print("Failed to load data. Aborting storage process.")
        return

    print("Beginning to stream records from file...")
    print(f"Preparing multi-vectors for ChromaDB collection '{collection.name}'...")

    # Batch size for adding data to ChromaDB
    batch_size = 100 
    ids_batch = []
    embeddings_batch = []
    metadatas_batch = []
    documents_batch = []
    total_vectors_processed = 0
    total_cases_processed = 0

    for i, case in enumerate(data):
        try:
            case_id = case.get("url") or case.get("medium_neutral_citation") or f"case_{i}"

            if not case_id:
                print(f"Skipping case {i+1}: Could not generate a case ID.")
                continue

            paragraphs = case.get("paragraphs")

            print(f"Processing case: {case.get('medium_neutral_citation', case_id)}")

            # --- Create and Embed Vectors for Specific Metadata Fields ---
            for field in METADATA_FIELDS_TO_VECTORIZE:
                field_value = case.get(field)

                # Handle list values by joining them for embedding
                if isinstance(field_value, list):
                    # Join list items into a single string for embedding
                    field_value_text = ", ".join(map(lambda x: str(x) if not isinstance(x, dict) else x.get("text", str(x)), field_value))
                elif field_value is not None:
                    field_value_text = str(field_value)
                else:
                    field_value_text = "" 

                # Only create a vector if there's actual text content
                if field_value_text.strip():
                    try:
                        meta_field_vector = embedding_model.encode(field_value_text).tolist()

                        # Metadata for this specific metadata field vector
                        meta_field_chroma_metadata: Dict[str, Any] = {
                            "type": f"metadata_{field}", # Unique type for each metadata field
                            "original_case_id": str(case_id),
                            "field_name": field,
                            "field_value_text": field_value_text, # Store the text that was embedded
                            "medium_neutral_citation": case.get("medium_neutral_citation", ""),
                            "url": case.get("url", ""),
                            "category": case.get("category", ""),
                        }
                        # Ensure ChromaDB compatibility for metadata values
                        for key, val in list(meta_field_chroma_metadata.items()): 
                            if isinstance(val, dict): 
                                try:
                                    meta_field_chroma_metadata[key] = json.dumps(val) 
                                except Exception:
                                    print(f"Warning: Could not JSON stringify metadata key '{key}' for case {case_id}, field '{field}'. Removing.")
                                    del meta_field_chroma_metadata[key]
                            elif val is None:
                                meta_field_chroma_metadata[key] = "" 

                        # Unique ID for this metadata field vector
                        meta_field_vector_id = f"{case_id}_meta_{field}"

                        # Append to batches
                        ids_batch.append(meta_field_vector_id)
                        embeddings_batch.append(meta_field_vector)
                        metadatas_batch.append(meta_field_chroma_metadata)
                        documents_batch.append(field_value_text)
                        total_vectors_processed += 1

                    except Exception as meta_emb_e:
                        print(f"Warning: Could not embed or prep metadata field '{field}' for case {case_id}: {meta_emb_e}")

            # --- Create and Embed Vectors for Paragraphs ---
            if paragraphs and isinstance(paragraphs, list):
                for paragraph_data in paragraphs:
                    p_num = paragraph_data.get("p_num")
                    p_text = paragraph_data.get("text")

                    if p_text is None or not p_text.strip():
                        # print(f"Skipping empty paragraph in case {case_id} (p_num: {p_num}).")
                        continue

                    # Unique ID for the paragraph vector
                    paragraph_vector_id = f"{case_id}_p{p_num}" if p_num is not None else f"{case_id}_pUnknown_{total_vectors_processed}"

                    try:
                        paragraph_vector = embedding_model.encode(p_text).tolist()

                        # Metadata for the paragraph vector
                        paragraph_chroma_metadata: Dict[str, Any] = {
                            "type": "paragraph",
                            "original_case_id": str(case_id), 
                            "p_num": p_num if p_num is not None else -1, 
                            "text": p_text, 
                            "medium_neutral_citation": case.get("medium_neutral_citation", ""),
                            "url": case.get("url", ""),
                            "category": case.get("category", ""),
                            
                        }
                        for key, val in list(paragraph_chroma_metadata.items()):
                            if isinstance(val, dict):
                                try:
                                    paragraph_chroma_metadata[key] = json.dumps(val)
                                except Exception:
                                    print(f"Warning: Could not JSON stringify metadata key '{key}' for paragraph {paragraph_vector_id}. Removing.")
                                    del paragraph_chroma_metadata[key]
                            elif val is None:
                                 paragraph_chroma_metadata[key] = "" 
                            elif key == "p_num" and val == "":
                                paragraph_chroma_metadata[key] = -1 


                        # Append to batches
                        ids_batch.append(paragraph_vector_id)
                        embeddings_batch.append(paragraph_vector)
                        metadatas_batch.append(paragraph_chroma_metadata)
                        documents_batch.append(p_text) 
                        total_vectors_processed += 1

                    except Exception as p_emb_e:
                        print(f"Error embedding or prepping paragraph {p_num} for case {case_id}: {p_emb_e}. Skipping paragraph.")
                        continue
            else:
                if "paragraphs" in case:
                     print(f"Skipping paragraphs for case {case_id}: Invalid 'paragraphs' field format (expected a list).")

            # --- Add batch to ChromaDB if ready ---
            if len(ids_batch) >= batch_size:
                print(f"Adding batch of {len(ids_batch)} vectors to ChromaDB (total processed: {total_vectors_processed})...")
                try:

                    collection.add(
                        ids=ids_batch,
                        embeddings=embeddings_batch,
                        metadatas=metadatas_batch,
                        documents=documents_batch
                    )
                    print(f"Batch added.")
                    # Reset batches
                    ids_batch = []
                    embeddings_batch = []
                    metadatas_batch = []
                    documents_batch = []
                except Exception as add_e:
                    print(f"ERROR during ChromaDB add batch (total processed before batch: {total_vectors_processed - len(ids_batch)}): {add_e}")
                    # For simplicity, clear the batch and continue
                    ids_batch = []
                    embeddings_batch = []
                    metadatas_batch = []
                    documents_batch = []

            total_cases_processed += 1

        except Exception as case_e:
            print(f"Severe Error processing case {i+1} (ID: {case.get('medium_neutral_citation', 'N/A')}): {case_e}")
            continue 

    # --- Add any remaining vectors in the last batch ---
    if ids_batch:
        print(f"Adding final batch of {len(ids_batch)} vectors to ChromaDB (total processed: {total_vectors_processed})...")
        try:
            collection.add(
                ids=ids_batch,
                embeddings=embeddings_batch,
                metadatas=metadatas_batch,
                documents=documents_batch 
            )
            print(f"Final batch added.")
        except Exception as add_e:
            print(f"ERROR during final ChromaDB add batch: {add_e}")

    print(f"\nFinished processing file.")
    print(f"Total cases processed: {total_cases_processed}")
    print(f"Total vectors added/attempted: {total_vectors_processed}")
    print(f"Current count in ChromaDB collection '{collection.name}': {collection.count()}")


# --- Main Execution ---
print(f"Starting ingestion process for file: {file_path}")
start_time = time.time()
store_case_multi_vectors_in_chroma(file_path, collection, embedding_model)
end_time = time.time()
print(f"Ingestion process finished in {end_time - start_time:.2f} seconds.")
print("ChromaDB data persisted.")