# indexer_qdrant/indexer_qdrant.py
import json
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, Range
from typing import List, Dict, Any, Union
import os
from dotenv import load_dotenv
import time
import uuid # Import the uuid library

load_dotenv()

# --- Qdrant Configuration ---
# Connect to the Qdrant service defined in docker-compose.yml
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost") # Use environment variable, fallback to localhost
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))   # Use environment variable, fallback to 6333

print(f"Attempting to connect to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
try:
    # Use the client to connect to the running Qdrant service
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    # Optional: check connection
    qdrant_client.get_collections()
    print("Connected to Qdrant service.")
except Exception as e:
    print(f"Error connecting to Qdrant service at {QDRANT_HOST}:{QDRANT_PORT}: {e}")
    print("Please ensure the Qdrant service is running and accessible.")
    exit() # Exit if connection fails

collection_name = "law_rag_qdrant"
# BAAI/bge-large-en-v1.5 outputs 1024 dimensions.
vector_dimension = 1024
sentence_transformer_model_name = "BAAI/bge-large-en-v1.5"
file_path = r"/app/data/cases_2.jsonl" # Adjust if your data path changes

# Define a namespace UUID for generating deterministic UUID v5 IDs
# This can be any valid UUID. A common practice is to generate one once.
# Example namespace (replace with your own if preferred, but keep it consistent)
RAG_NAMESPACE_UUID = uuid.UUID('f9a9f8e0-6b0d-4e4b-8c9c-8c9c8c9c8c9c')


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

# Helper function to create a stable UUID v5 ID
def generate_qdrant_uuid5(input_string: str) -> str:
    """Generates a stable, valid Qdrant UUID v5 string ID."""
    # uuid.uuid5 requires a namespace UUID and the name (input string)
    return str(uuid.uuid5(RAG_NAMESPACE_UUID, input_string))


def store_case_multi_vectors_in_qdrant(file_path: str, qdrant_client: QdrantClient, collection_name: str, embedding_model):
    """
    Loads case data, generates embeddings using SentenceTransformer,
    and stores them in a Qdrant collection with associated metadata.
    Creates separate vectors for specified metadata fields and paragraphs.
    """
    print(f"Loading data from {file_path}...")
    data = load_data(file_path)

    if data is None:
        print("Failed to load data. Aborting storage process.")
        return

    # Ensure collection exists
    try:
        print(f"Ensuring Qdrant collection '{collection_name}' exists...")
        # Use recreate_collection to start fresh or get existing
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dimension, distance=Distance.COSINE),
        )
        print(f"Collection '{collection_name}' ready.")
    except Exception as e:
        print(f"Error ensuring Qdrant collection exists: {e}")
        raise

    print("Beginning to stream records from file...")
    print(f"Preparing multi-vectors for Qdrant collection '{collection_name}'...")

    # Batch size for adding data to Qdrant
    batch_size = 100
    points_batch = []
    total_vectors_processed = 0
    total_cases_processed = 0

    for i, case in enumerate(data):
        try:
            # Using URL as a unique identifier, fallback to citation or index
            # Ensure case_id_base is something reasonably stable and unique
            case_id_base = case.get("url") or case.get("medium_neutral_citation") or f"case_index_{i}"
            # Convert case_id_base to a string for hashing
            case_id_str = str(case_id_base)


            if not case_id_str:
                print(f"Skipping case {i+1}: Could not generate a case ID string.")
                continue

            paragraphs = case.get("paragraphs")

            print(f"Processing case: {case.get('medium_neutral_citation', case_id_base)}")

            # --- Create and Embed Vectors for Specific Metadata Fields ---
            for field in METADATA_FIELDS_TO_VECTORIZE:
                field_value = case.get(field)

                # Handle list values by joining them for embedding
                if isinstance(field_value, list):
                    field_value_text = ", ".join(map(lambda x: str(x) if not isinstance(x, dict) else x.get("text", str(x)), field_value))
                elif field_value is not None:
                    field_value_text = str(field_value)
                else:
                    field_value_text = ""

                if field_value_text.strip():
                    try:
                        meta_field_vector = embedding_model.encode(field_value_text).tolist()

                        # Metadata for this specific metadata field vector
                        meta_field_payload: Dict[str, Any] = {
                            "type": f"metadata_{field}", # Unique type for each metadata field
                            "original_case_id": case_id_str, # Store the base case ID string (original format)
                            "field_name": field,
                            "field_value_text": field_value_text, # Store the text that was embedded
                            "medium_neutral_citation": case.get("medium_neutral_citation", ""),
                            "url": case.get("url", ""),
                            "category": case.get("category", ""),
                        }
                        # Ensure Qdrant compatibility for payload values (simple types)
                        for key, val in list(meta_field_payload.items()):
                            if isinstance(val, dict):
                                # print(f"Warning: Payload key '{key}' for case {case_id_str}, field '{field}' is a dict. Qdrant supports simple types. Storing as string.")
                                try:
                                    meta_field_payload[key] = json.dumps(val)
                                except Exception:
                                     print(f"Warning: Could not JSON stringify payload key '{key}' for case {case_id_str}, field '{field}'. Removing.")
                                     del meta_field_payload[key]

                            elif val is None:
                                meta_field_payload[key] = ""

                        # Generate a stable UUID v5 ID from the combination of case ID and field
                        # Combine case_id_str and field to ensure uniqueness per metadata field per case
                        raw_id_string = f"{case_id_str}_meta_{field}"
                        meta_field_vector_id = generate_qdrant_uuid5(raw_id_string)


                        points_batch.append(PointStruct(
                            id=meta_field_vector_id, # Use the UUID v5 as the point ID
                            vector=meta_field_vector,
                            payload=meta_field_payload
                        ))
                        total_vectors_processed += 1

                    except Exception as meta_emb_e:
                        print(f"Warning: Could not embed or prep metadata field '{field}' for case {case_id_str}: {meta_emb_e}")

            # --- Create and Embed Vectors for Paragraphs ---
            if paragraphs and isinstance(paragraphs, list):
                for paragraph_data in paragraphs:
                    p_num = paragraph_data.get("p_num")
                    p_text = paragraph_data.get("text")

                    if p_text is None or not p_text.strip():
                        continue

                    # Generate a stable UUID v5 ID from the combination of case ID and paragraph number
                    # Combine case_id_str and p_num to ensure uniqueness per paragraph per case
                    # Handle p_num being None or empty string for ID generation consistency
                    p_num_str = str(p_num) if p_num is not None and p_num != "" else "Unknown"
                    raw_id_string = f"{case_id_str}_p{p_num_str}"
                    paragraph_vector_id = generate_qdrant_uuid5(raw_id_string)


                    try:
                        paragraph_vector = embedding_model.encode(p_text).tolist()

                        # Metadata for the paragraph vector
                        paragraph_payload: Dict[str, Any] = {
                            "type": "paragraph",
                            "original_case_id": case_id_str, # Store the base case ID string (original format)
                            "p_num": p_num if p_num is not None else -1,
                            "text": p_text,
                            "medium_neutral_citation": case.get("medium_neutral_citation", ""),
                            "url": case.get("url", ""),
                            "category": case.get("category", ""),

                        }
                         # Ensure Qdrant compatibility for payload values
                        for key, val in list(paragraph_payload.items()):
                            if isinstance(val, dict):
                                # print(f"Warning: Payload key '{key}' for paragraph {paragraph_vector_id} is a dict. Qdrant supports simple types. Storing as string.")
                                try:
                                    paragraph_payload[key] = json.dumps(val)
                                except Exception:
                                     print(f"Warning: Could not JSON stringify payload key '{key}' for paragraph {paragraph_vector_id}. Removing.")
                                     del paragraph_payload[key]
                            elif val is None:
                                 paragraph_payload[key] = ""
                            elif key == "p_num" and val == "":
                                paragraph_payload[key] = -1


                        points_batch.append(PointStruct(
                            id=paragraph_vector_id, # Use the UUID v5 as the point ID
                            vector=paragraph_vector,
                            payload=paragraph_payload
                        ))
                        total_vectors_processed += 1

                    except Exception as p_emb_e:
                        print(f"Error embedding or prepping paragraph {p_num} for case {case_id_str}: {p_emb_e}. Skipping paragraph.")
                        continue
            else:
                if "paragraphs" in case:
                     print(f"Skipping paragraphs for case {case_id_str}: Invalid 'paragraphs' field format (expected a list).")


            # --- Upload batch to Qdrant if ready ---
            if len(points_batch) >= batch_size:
                print(f"Adding batch of {len(points_batch)} vectors to Qdrant (total processed: {total_vectors_processed})...")
                try:
                    qdrant_client.upsert(
                        collection_name=collection_name,
                        wait=True,
                        points=points_batch
                    )
                    print(f"Batch added.")
                    # Reset batches
                    points_batch = []
                except Exception as upsert_e:
                    print(f"ERROR during Qdrant upsert batch (total processed before batch: {total_vectors_processed - len(points_batch)}): {upsert_e}. Clearing batch and attempting to continue...")
                    # Clear the batch to avoid re-attempting the problematic points indefinitely
                    points_batch = []


            total_cases_processed += 1

        except Exception as case_e:
            print(f"Severe Error processing case {i+1} (ID: {case.get('medium_neutral_citation', case_id_str if 'case_id_str' in locals() else 'N/A')}): {case_e}")
            continue

    # --- Add any remaining vectors in the last batch ---
    if points_batch:
        print(f"Adding final batch of {len(points_batch)} vectors to Qdrant (total processed: {total_vectors_processed})...")
        try:
            qdrant_client.upsert(
                collection_name=collection_name,
                wait=True,
                points=points_batch
            )
            print(f"Final batch added.")
        except Exception as upsert_e:
            print(f"ERROR during final Qdrant upsert batch: {upsert_e}")

    print(f"\nFinished processing file.")
    print(f"Total cases processed: {total_cases_processed}")
    print(f"Total vectors added/attempted: {total_vectors_processed}")
    # You can get the actual count from the collection info if needed after upsert completes
    try:
        collection_info = qdrant_client.get_collection(collection_name=collection_name)
        print(f"Current count in Qdrant collection '{collection_name}': {collection_info.vectors_count}")
    except Exception as count_e:
        print(f"Could not retrieve collection count: {count_e}")


    print("Qdrant indexing process completed.")


# --- Main Execution ---
print(f"Starting ingestion process for file: {file_path}")
start_time = time.time()
store_case_multi_vectors_in_qdrant(file_path, qdrant_client, collection_name, embedding_model)
end_time = time.time()
print(f"Ingestion process finished in {end_time - start_time:.2f} seconds.")