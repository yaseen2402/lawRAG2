import os
from dotenv import load_dotenv
import chromadb
import time
import json
from typing import Set, Optional, List, Tuple, Dict

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
collection_name = "law_rag"
persist_directory = r"C:\Users\hp\lawRAG2\chroma_db_storage" # ** ADJUST AS NEEDED **

# --- Configuration for JSONL Comparison ---
# ** SET THE CORRECT PATH to your JSONL file **
jsonl_file_path = r"C:\Users\hp\lawRAG2\data\cases_2.jsonl"
# ** SET THE KEY in your JSONL that holds the case ID **
json_id_key = "url" # As specified: "key url and value as the case id"

# --- Function to Get Unique IDs from ChromaDB ---
def get_unique_case_ids_from_chroma(
    collection_name: str,
    persist_path: str
) -> Optional[Set[str]]:
    """
    Connects to a persistent ChromaDB collection and retrieves all unique
    'original_case_id' values from the document metadata.
    (Implementation is the same as the previous script)

    Returns:
        A set containing all unique 'original_case_id' strings found,
        or None if an error occurs. Returns an empty set if collection is empty.
    """
    print("-" * 30)
    print("Starting script to fetch unique case IDs from ChromaDB...")
    print(f"Target Collection: '{collection_name}'")
    print(f"Persist Directory: '{persist_path}'")
    print("-" * 30)

    if not os.path.exists(persist_path):
        print(f"Error: Persist directory not found at '{persist_path}'.")
        return None

    print("Initializing ChromaDB client...")
    try:
        client = chromadb.PersistentClient(path=persist_path)
        print("ChromaDB client initialized.")
    except Exception as e:
        print(f"Error initializing ChromaDB client: {e}")
        return None

    print(f"Attempting to get collection: '{collection_name}'...")
    try:
        collection = client.get_collection(name=collection_name)
        collection_count = collection.count()
        print(f"Successfully connected to collection '{collection.name}'.")
        print(f"Total items in collection: {collection_count}")
        if collection_count == 0:
            print("ChromaDB collection is empty.")
            return set()
    except Exception as e:
        print(f"Error getting collection '{collection_name}': {e}")
        return None

    print("Fetching all document metadata from ChromaDB...")
    start_time = time.time()
    try:
        results = collection.get(include=['metadatas'])
        end_time = time.time()
        print(f"Metadata retrieval finished in {end_time - start_time:.2f} seconds.")
    except Exception as e:
        print(f"Error fetching data from ChromaDB collection: {e}")
        return None

    unique_case_ids: Set[str] = set()
    retrieved_metadatas = results.get('metadatas')

    if not retrieved_metadatas:
        print("No metadata found in the retrieved ChromaDB results.")
        return set()

    print(f"Processing metadata for {len(retrieved_metadatas)} items...")
    for metadata in retrieved_metadatas:
        if metadata:
            case_id = metadata.get('original_case_id')
            if case_id:
                unique_case_ids.add(str(case_id).strip()) # Ensure string and strip whitespace

    print(f"Found {len(unique_case_ids)} unique 'original_case_id' values in ChromaDB.")
    return unique_case_ids

# --- Function to Compare with JSONL File ---
def find_missing_ids_in_jsonl(
    chroma_ids: Set[str],
    jsonl_filepath: str,
    id_key: str
) -> List[Tuple[int, str]]:
    """
    Reads a JSONL file, extracts case IDs, and finds which ones are NOT
    present in the provided set of ChromaDB IDs.

    Args:
        chroma_ids: A set of case IDs retrieved from ChromaDB.
        jsonl_filepath: Path to the JSONL file.
        id_key: The key in each JSON line that contains the case ID.

    Returns:
        A list of tuples, where each tuple contains (line_number, missing_case_id).
        Returns an empty list if the file doesn't exist or no IDs are missing.
    """
    print("-" * 30)
    print(f"Comparing ChromaDB IDs with JSONL file: '{jsonl_filepath}'")
    print(f"Using key '{id_key}' to find case IDs in JSONL.")
    print("-" * 30)

    missing_ids_info: List[Tuple[int, str]] = []

    if not os.path.exists(jsonl_filepath):
        print(f"Error: JSONL file not found at '{jsonl_filepath}'")
        return missing_ids_info # Return empty list

    line_number = 0
    found_json_ids = 0
    try:
        with open(jsonl_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line_number += 1
                line_content = line.strip()
                if not line_content: # Skip empty lines
                    continue

                try:
                    data = json.loads(line_content)
                    case_id_from_file = data.get(id_key)

                    if case_id_from_file is not None:
                        # Ensure it's a string and strip whitespace for reliable comparison
                        case_id_from_file_str = str(case_id_from_file).strip()
                        found_json_ids += 1
                        # Check if this ID is missing from the set retrieved from ChromaDB
                        if case_id_from_file_str not in chroma_ids:
                            missing_ids_info.append((line_number, case_id_from_file_str))
                    else:
                        print(f"Warning: Key '{id_key}' not found or is null on line {line_number}.")

                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON on line {line_number}. Content: '{line_content[:100]}...'")
                except Exception as e:
                    print(f"Warning: Error processing line {line_number}: {e}")

    except IOError as e:
        print(f"Error reading file '{jsonl_filepath}': {e}")
        return [] # Return empty list on file read error

    print(f"Processed {line_number} lines from JSONL file.")
    print(f"Found {found_json_ids} valid case IDs using key '{id_key}'.")
    return missing_ids_info

# --- Execution ---
if __name__ == "__main__":
    # 1. Get unique IDs from ChromaDB
    chroma_case_ids = get_unique_case_ids_from_chroma(collection_name, persist_directory)

    if chroma_case_ids is None:
        print("\nFailed to retrieve IDs from ChromaDB. Cannot proceed with comparison.")
    else:
        # 2. Compare with JSONL file
        missing_from_chroma = find_missing_ids_in_jsonl(
            chroma_ids=chroma_case_ids,
            jsonl_filepath=jsonl_file_path,
            id_key=json_id_key
        )

        print("-" * 30)
        # 3. Report results
        if not missing_from_chroma:
            # Check if the file was actually processed
            if os.path.exists(jsonl_file_path):
                 print("Comparison complete. All case IDs found in the JSONL file are present in the ChromaDB collection.")
            else:
                 print("Comparison could not be completed because the JSONL file was not found.")
        else:
            print(f"Found {len(missing_from_chroma)} case IDs from the JSONL file that are NOT in the ChromaDB collection:")
            for line_num, missing_id in missing_from_chroma:
                print(f"  - Line {line_num}: '{missing_id}'")
        print("-" * 30)