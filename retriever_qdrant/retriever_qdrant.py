# retriever_qdrant/retriever_qdrant.py
print("DEBUG: Script started.") # ADD THIS AS THE VERY FIRST LINE

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Filter, FieldCondition, Range, MatchAny
from sentence_transformers import SentenceTransformer
import time
from typing import List, Dict, Any, Union
import gradio as gr

from langchain_google_genai import ChatGoogleGenerativeAI
# We will use Qdrant client directly for flexibility, not Langchain's Qdrant wrapper
# from langchain_community.vectorstores import Qdrant # Optional, but direct client offers more control
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Qdrant Configuration ---
# Configure connection to your Qdrant instance
# If running Qdrant as a separate service via Docker, use its host/port
# For this example, assuming it's running on localhost
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
collection_name = "law_rag_qdrant"

# Embedding Model Configuration
embedding_model_name = "BAAI/bge-large-en-v1.5"
vector_dimension = 1024

# LLM Configuration
llm_model_name = "gemini-1.5-flash-latest"

# Retrieval Configuration
INITIAL_SEARCH_K = 5
MAX_FILTERED_K = 200
TOTAL_FILTER_QUERY_LIMIT = 1000


# Initialize Qdrant Client
print(f"Initializing Qdrant client for {QDRANT_HOST}:{QDRANT_PORT}")
try:
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print(qdrant_client.count("law_rag_qdrant", exact=True))
    # Check if collection exists
    collection_info = qdrant_client.get_collection(collection_name=collection_name)
    print(f"Connected to Qdrant collection '{collection_name}'. Total items: {collection_info.status}")
    if collection_info.vectors_count == 0:
         print("Warning: Collection is empty. Please run the Qdrant indexing script first.")
except Exception as e:
    print(f"Error initializing Qdrant client or connecting to collection: {e}")
    print(f"Ensure Qdrant is running at {QDRANT_HOST}:{QDRANT_PORT} and the collection '{collection_name}' exists.")
    exit()


# Initialize the Sentence Transformer embedding model
print(f"Loading Sentence Transformer model: {embedding_model_name}")
try:
    # Using SentenceTransformer directly for embedding queries
    embedding_model = SentenceTransformer(embedding_model_name)
    print("Sentence Transformer model loaded successfully.")
except Exception as e:
    print(f"Error loading Sentence Transformer model {embedding_model_name}: {e}")
    print("Install dependencies: `pip install sentence-transformers torch`.")
    exit()


# Initialize the LLM
print(f"Initializing LLM: {llm_model_name}")
try:
    llm = ChatGoogleGenerativeAI(model=llm_model_name, google_api_key=GOOGLE_API_KEY)
    print("LLM initialized.")
except Exception as e:
    print(f"Error initializing LLM: {e}")
    print("Ensure GOOGLE_API_KEY is set in .env file.")
    exit()

# --- Custom Enhanced Retrieval Function using Qdrant Client ---

def enhanced_retrieval_function_qdrant(query: str) -> List[Document]:
    """
    Performs initial similarity search using Qdrant, identifies case IDs,
    retrieves all chunks for those cases from Qdrant, and returns unique documents.
    """
    print(f"Performing initial similarity search for query: '{query}' (k={INITIAL_SEARCH_K})")
    start_time_retrieval = time.time()

    try:
        # Embed the query using the same model as indexing
        query_vector = embedding_model.encode(query).tolist()

        # Step 1: Perform initial similarity search in Qdrant
        search_result = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=INITIAL_SEARCH_K,
            with_payload=True, # Include payload (metadata)
        )
        print(f"Initial Qdrant search found {len(search_result.points)} results.")
    except Exception as e:
        print(f"Error during initial Qdrant search: {e}")
        return []

    if not search_result:
        print("No initial relevant documents found.")
        return []

    # Step 2: Extract unique original_case_ids from the initial results
    case_ids = set()
    print("Identifying case IDs from initial results...")
    for hit in search_result.points:
        # Access payload (metadata)
        case_id = hit.payload.get('original_case_id')
        print(f"  - Initial hit ID {hit.id}, case_id: {case_id}")
        if case_id:
            case_ids.add(case_id)
    print(f"[Retrieval] Unique case IDs: {case_ids}")

    if not case_ids:
        print("No case IDs found in initial results payload.")
        # If no case IDs, just return the initial search results converted to Documents
        print("Returning initial documents as context.")
        return [Document(page_content=hit.payload.get('text', ''), metadata=hit.payload) for hit in search_result]


    # Step 3: Retrieve all chunks belonging to these case IDs using a metadata filter
    try:
        # Construct the filter for case IDs
        case_id_filter = Filter(
            must=[
                FieldCondition(
                    key="original_case_id",
                    match=MatchAny(any=list(case_ids))
                )
            ]
        )

        # Use scroll API to retrieve all points matching the filter up to the limit
        # Note: Scroll is for retrieval, not ordered search. Limit applies to total.
        scroll_result, _next_page_offset = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=case_id_filter,
            limit=TOTAL_FILTER_QUERY_LIMIT,
            with_payload=True,
            with_vectors=False # No need to retrieve vectors for context documents
        )
        print(f"Retrieved {len(scroll_result)} chunks via filter (scroll).")


    except Exception as e:
        print(f"Error during filtered retrieval (scroll) from Qdrant: {e}")
        print("Falling back to returning initial documents as context due to filter error.")
        # Convert initial search results to Langchain Documents as a fallback
        return [Document(page_content=hit.payload.get('text', ''), metadata=hit.payload) for hit in search_result]

    counts = {}
    for hit in scroll_result:
        cid = hit.payload.get('original_case_id')
        counts[cid] = counts.get(cid, 0) + 1
    print("[Retrieval] Chunk counts by case:")
    for cid, cnt in counts.items():
        print(f"  - Case {cid}: {cnt} chunks")

    # Step 4: Combine and Deduplicate all retrieved documents
    # Combine initial search hits (for scoring context) and filtered scroll results
    all_docs_map = {}
    print("Combining and deduplicating documents...")

    # Add initial search hits to map
    for hit in search_result.points:
         # Qdrant hit ID can be used as a unique key
         all_docs_map[hit.id] = Document(page_content=hit.payload.get('text', ''), metadata=hit.payload)

    # Add filtered scroll results to map, using hit ID for deduplication
    for hit in scroll_result:
        all_docs_map[hit.id] = Document(page_content=hit.payload.get('text', ''), metadata=hit.payload)


    # Convert map values back to a list of Documents
    combined_docs = list(all_docs_map.values())
    print(f"Combined and unique documents for context: {len(combined_docs)}")
    end_time_retrieval = time.time()
    print(f"Retrieval step finished in {end_time_retrieval - start_time_retrieval:.2f} seconds.")

    return combined_docs

# --- Set up the RAG Chain using LCEL ---

qa_system_prompt = """You are a legal assistant. Use the following court case information to answer the user's question.
If the information does not contain the answer, state that you cannot find the answer in the provided cases.
Do not make up information. Cite the source for each piece of information you use from the 'Relevant court case information'. The source is available in the metadata, typically under 'medium_neutral_citation' or 'original_case_id'. If a specific paragraph number ('p_num') is available, also include that.

Relevant court case information:
{context}

User question: {input}
Legal Assistant Answer:
"""
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    ("human", "{input}"),
])
print("QA Prompt template created.")


# Use the custom Qdrant retrieval function in the RAG chain
rag_chain = (
    {"context": enhanced_retrieval_function_qdrant, "input": RunnablePassthrough()}
    | qa_prompt
    | llm
    | StrOutputParser()
)
print("RAG chain created using LCEL with custom Qdrant retrieval.")


def answer_question_with_sources_qdrant(user_query: str) -> str:
    print(f"DEBUG: Function answer_question_with_sources_qdrant called with query: '{user_query}'") # ADD THIS LINE
    if not user_query.strip():
        return "Please enter a valid question."

    try:
        response_text = rag_chain.invoke(user_query)

        print("\n--- Sources (Retrieved Full Case Context) ---")

        # Re-run retrieval to get source documents for listing
        source_docs = enhanced_retrieval_function_qdrant(user_query)
        sources = set()
        for doc in source_docs:
            title = doc.metadata.get('medium_neutral_citation') or doc.metadata.get('original_case_id', 'Unknown Case')
            url = doc.metadata.get('original_case_id') # Using original_case_id as a placeholder for URL
            if url:
                 sources.add(f"[{title}]({url})")
            else:
                 sources.add(title)


        sources_text = "\n".join(f"- {s}" for s in sorted(sources)) or "No sources found."
        return f"### Answer:\n{response_text}\n\n---\n### Sources:\n{sources_text}"

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        return f"An error occurred: {e}"

# Gradio Interface
gr.Interface(
    fn=answer_question_with_sources_qdrant,
    inputs=gr.Textbox(lines=3, placeholder="Ask your legal question here..."),
    outputs=gr.Markdown(),
    title="Legal RAG System (Qdrant)",
    description="Ask legal questions and get answers grounded in retrieved court case documents from Qdrant.",
    allow_flagging="never"
).launch(server_name="0.0.0.0")