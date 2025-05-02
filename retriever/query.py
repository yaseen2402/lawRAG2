# A more advanced retrieval strategy often called "Parent Document Retrieval" or "Contextual Compression with Re-ranking" conceptually, where you use similarity to find initial relevant points, but then expand the context around those points (in your case, expanding to the full case).

import os
from dotenv import load_dotenv
# Removed Pinecone imports
# import pinecone
# from pinecone import Pinecone
import chromadb # Import ChromaDB
from sentence_transformers import SentenceTransformer
import time
from typing import List, Dict, Any, Union
import gradio as gr

# Langchain components
from langchain_google_genai import ChatGoogleGenerativeAI
# Update imports for Langchain v0.2+ community package
from langchain_community.vectorstores import Chroma # Use Langchain's Chroma integration
from langchain_community.embeddings import SentenceTransformerEmbeddings # Langchain wrapper for ST
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document # To work with Langchain Document objects

# Load environment variables
load_dotenv()

# API Keys and Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- ChromaDB Configuration ---
collection_name = "law_rag" # Name for your ChromaDB collection (must match indexer)
persist_directory = "/app/chroma_db_storage" # Points to lawRAG/chroma_db_storage# Ensure the directory exists (though the ingest script should have created it)
os.makedirs(persist_directory, exist_ok=True)

# Embedding Model Configuration
embedding_model_name = "BAAI/bge-large-en-v1.5"
dimension = 1024 # Must match the model dimension and collection dimension

# LLM Configuration
llm_model_name = "gemini-1.5-flash-latest" # Or "gemini-pro"

# Retrieval Configuration
INITIAL_SEARCH_K = 5 # Number of initial chunks to retrieve via similarity search
MAX_FILTERED_K = 200 # Maximum number of chunks to retrieve per case after filtering (set high enough, cases might have >100 paragraphs)
# Set a reasonable total limit for filtered query to avoid overwhelming Chroma/memory
TOTAL_FILTER_QUERY_LIMIT = 1000 # Limit the total number of items fetched in the filtered query


# Initialize ChromaDB Client
print(f"Initializing ChromaDB client from: {persist_directory}")
try:
    client = chromadb.PersistentClient(path=persist_directory)
    print("ChromaDB client initialized.")
except Exception as e:
    print(f"Error initializing ChromaDB client: {e}")
    exit()

# Get or connect to the ChromaDB Collection
try:
    print(f"Getting ChromaDB collection: '{collection_name}'...")
    collection = client.get_collection(name=collection_name)
    print(f"Connected to collection '{collection.name}'. Total items: {collection.count()}")
    if collection.count() == 0:
         print("Warning: Collection is empty. Please run the indexing script first.")
         # Decide if you want to exit or allow searching an empty collection
         # exit()
except Exception as e:
    print(f"Error connecting to ChromaDB collection '{collection_name}': {e}")
    print("Ensure the collection was created by the indexing script.")
    exit()


# Initialize the Sentence Transformer embedding model
print(f"Loading Sentence Transformer model: {embedding_model_name}")
try:
    # Using the Langchain wrapper is convenient for vector store integration
    embedding_function = SentenceTransformerEmbeddings(model_name=embedding_model_name)
    # We also load it directly for potential use in the retrieval function if needed,
    # but the Langchain wrapper handles embedding for similarity search.
    # direct_embedding_model = SentenceTransformer(embedding_model_name) # Keep if needed for manual embedding
    print("Sentence Transformer model loaded successfully.")
except Exception as e:
    print(f"Error loading Sentence Transformer model {embedding_model_name}: {e}")
    print("Install dependencies: `pip install sentence-transformers torch langchain-community`.")
    exit()

# Initialize Langchain's Chroma Vector Store wrapper
# This wrapper is useful for performing the initial similarity search using the embedding_function
print("Initializing Langchain Chroma vector store...")
vectorstore = Chroma(
    client=client,
    collection_name=collection_name,
    embedding_function=embedding_function, # Pass the Langchain Embedding object
    persist_directory=persist_directory # Optional, but good practice
)
print("Langchain Chroma vector store initialized.")


# Initialize the LLM
print(f"Initializing LLM: {llm_model_name}")
try:
    llm = ChatGoogleGenerativeAI(model=llm_model_name, google_api_key=GOOGLE_API_KEY)
    print("LLM initialized.")
except Exception as e:
    print(f"Error initializing LLM: {e}")
    print("Ensure GOOGLE_API_KEY is set in .env file.")
    exit()

# --- Custom Enhanced Retrieval Function ---

def enhanced_retrieval_function(query: str) -> List[Document]:
    """
    Performs initial similarity search using Chroma, identifies case IDs,
    retrieves all chunks for those cases from Chroma, and returns unique documents.
    """
    print(f"Performing initial similarity search for query: '{query}' (k={INITIAL_SEARCH_K})")
    start_time_retrieval = time.time()

    # Step 1: Perform initial similarity search using the Chroma vector store wrapper
    # This handles embedding the query and searching Chroma
    try:
        initial_docs = vectorstore.similarity_search(query, k=INITIAL_SEARCH_K)
        print(f"Initial similarity search found {len(initial_docs)} documents.")
    except Exception as e:
        print(f"Error during initial similarity search: {e}")
        return []


    if not initial_docs:
        print("No initial relevant documents found.")
        return []

    # Step 2: Extract unique original_case_ids from the initial results
    case_ids = set()
    print("Identifying case IDs from initial results...")
    for doc in initial_docs:
        # Access metadata from the Langchain Document object
        case_id = doc.metadata.get('original_case_id')
        if case_id:
            case_ids.add(case_id)

    if not case_ids:
        print("No case IDs found in initial results metadata.")
        # Return just the initial docs if no case IDs are found or metadata is missing
        print("Returning initial documents as context.")
        return initial_docs

    print(f"Found {len(case_ids)} unique case ID(s): {case_ids}")

    # Step 3: Retrieve all chunks belonging to these case IDs using a metadata filter
    # We use the direct Chroma client via the Langchain wrapper for a filter-only query
    print(f"Retrieving all chunks for identified cases (max_k per case = {MAX_FILTERED_K}, total limit={TOTAL_FILTER_QUERY_LIMIT})...")
    try:
        # Use Chroma's collection.query with a 'where' filter and no query vector/text
        # This effectively acts as a filtered get.
        # Include 'documents' and 'metadatas' to get text content and metadata back.
        # n_results must be large enough to potentially fetch all relevant chunks across cases.
        filter_results = collection.get(
            where={'original_case_id': {"$in": list(case_ids)}},
            include=['metadatas', 'documents'], # Important to get the text and metadata
            limit=TOTAL_FILTER_QUERY_LIMIT # Set a total limit
        )
        print(f"Retrieved {len(filter_results.get('ids', []))} chunks via filter.")

    except Exception as e:
        print(f"Error during filtered retrieval from ChromaDB: {e}")
        # If filtered retrieval fails, fall back to just initial docs
        print("Falling back to returning initial documents as context due to filter error.")
        return initial_docs

    # Step 4: Combine and Deduplicate all retrieved documents
    # Use a dictionary to easily handle deduplication based on the item ID
    all_docs_map = {}
    print("Combining and deduplicating documents...")

    # Add initial documents from similarity search (already in Langchain Document format)
    for doc in initial_docs:
        # Use the document's ID from Langchain (which should correspond to the Chroma item ID)
        # or a robust key if ID is not reliable. Assuming doc.metadata.get('id') or similar from indexing.
        # A safer key might be a combination of original_case_id and chunk type/identifier if ID isn't guaranteed unique/present.
        # Let's use the 'id' field from the metadata which should map to Chroma's item ID
        doc_id = doc.metadata.get('id') # Assuming 'id' is stored in metadata
        if not doc_id:
             # Fallback key if 'id' is not in metadata
             doc_id = f"{doc.metadata.get('original_case_id')}_{doc.metadata.get('type')}_{doc.metadata.get('p_num', '')}_{doc.metadata.get('field_name', '')}"
        if doc_id:
             all_docs_map[doc_id] = doc # Add initial docs

    # Add documents retrieved via filter (raw data from Chroma)
    # We need to convert raw Chroma results back to Langchain Document format
    if filter_results and filter_results.get('ids'):
        for i in range(len(filter_results['ids'])):
            item_id = filter_results['ids'][i]
            metadata = filter_results['metadatas'][i]
            doc_text = filter_results['documents'][i] # Text content from Chroma's 'documents' field

            if doc_text: # Only process if there is text content
                 # Create a new Langchain Document object
                doc = Document(page_content=doc_text, metadata=metadata)
                # Use the item ID from Chroma as the key for deduplication
                all_docs_map[item_id] = doc

    # Convert map values back to a list of Documents
    combined_docs = list(all_docs_map.values())
    print(f"Combined and unique documents for context: {len(combined_docs)}")
    end_time_retrieval = time.time()
    print(f"Retrieval step finished in {end_time_retrieval - start_time_retrieval:.2f} seconds.")

    return combined_docs

# --- Set up the RAG Chain using LCEL ---

# Define the prompt template for the LLM
# This remains the same as before, the context will be provided by our custom function
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

# Build the RAG chain using Langchain Expression Language (LCEL)
# This chain takes the user input, passes it to our custom retrieval function to get context,
# then passes the original input and the custom context to the prompt and the LLM.
rag_chain = (
    # Assign the result of enhanced_retrieval_function to the 'context' key
    # The original user input is passed through using RunnablePassthrough
    {"context": enhanced_retrieval_function, "input": RunnablePassthrough()}
    | qa_prompt # Pass {"context": ..., "input": ...} to the prompt
    | llm # Pass the formatted prompt to the LLM
    | StrOutputParser() # Parse the LLM's output into a string
)
print("RAG chain created using LCEL with custom retrieval.")


def answer_question_with_sources(user_query: str) -> str:

    if not user_query.strip():
        return "Please enter a valid question."

    try:
        
        response_text = rag_chain.invoke(user_query)
       
        print("\n--- Sources (Retrieved Full Case Context) ---")

        source_docs = enhanced_retrieval_function(user_query) # This re-runs the search and filter
        sources = set()
        for doc in source_docs:
            title = doc.metadata.get('medium_neutral_citation') or doc.metadata.get('original_case_id', 'Unknown Case')
            url = doc.metadata.get('original_case_id')  # Assuming you stored this in metadata during indexing
            if url:
                sources.add(f"[{title}]({url})")
            else:
                sources.add(title)

        sources_text = "\n".join(f"- {s}" for s in sorted(sources)) or "No sources found."
        return f"### Answer:\n{response_text}\n\n---\n### Sources:\n{sources_text}"
        

    except Exception as e:
        print(f"\nAn error occurred: {e}")

gr.Interface(
    fn=answer_question_with_sources,
    inputs=gr.Textbox(lines=3, placeholder="Ask your legal question here..."),
    outputs=gr.Markdown(),
    title="Legal RAG System",
    description="Ask legal questions and get answers grounded in retrieved court case documents.",
    allow_flagging="never"
).launch(server_name="0.0.0.0")