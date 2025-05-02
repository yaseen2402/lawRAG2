
import os
from dotenv import load_dotenv
import chromadb 
from sentence_transformers import SentenceTransformer
import time
from typing import List, Dict, Any, Union
import gradio as gr

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document 

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- ChromaDB Configuration ---
collection_name = "law_rag" 
persist_directory = "/app/chroma_db_storage" 
os.makedirs(persist_directory, exist_ok=True)

# Embedding Model Configuration
embedding_model_name = "BAAI/bge-large-en-v1.5"
dimension = 1024 

# LLM Configuration
llm_model_name = "gemini-1.5-flash-latest" 

# Retrieval Configuration
INITIAL_SEARCH_K = 5 
MAX_FILTERED_K = 200 
TOTAL_FILTER_QUERY_LIMIT = 1000 


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
except Exception as e:
    print(f"Error connecting to ChromaDB collection '{collection_name}': {e}")
    print("Ensure the collection was created by the indexing script.")
    exit()


# Initialize the Sentence Transformer embedding model
print(f"Loading Sentence Transformer model: {embedding_model_name}")
try:
    embedding_function = SentenceTransformerEmbeddings(model_name=embedding_model_name)
    print("Sentence Transformer model loaded successfully.")
except Exception as e:
    print(f"Error loading Sentence Transformer model {embedding_model_name}: {e}")
    print("Install dependencies: `pip install sentence-transformers torch langchain-community`.")
    exit()


print("Initializing Langchain Chroma vector store...")
vectorstore = Chroma(
    client=client,
    collection_name=collection_name,
    embedding_function=embedding_function, 
    persist_directory=persist_directory 
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
       
        filter_results = collection.get(
            where={'original_case_id': {"$in": list(case_ids)}},
            include=['metadatas', 'documents'], 
            limit=TOTAL_FILTER_QUERY_LIMIT 
        )
        print(f"Retrieved {len(filter_results.get('ids', []))} chunks via filter.")

    except Exception as e:
        print(f"Error during filtered retrieval from ChromaDB: {e}")
        print("Falling back to returning initial documents as context due to filter error.")
        return initial_docs

    # Step 4: Combine and Deduplicate all retrieved documents
    all_docs_map = {}
    print("Combining and deduplicating documents...")

    for doc in initial_docs:
        doc_id = doc.metadata.get('id') 
        if not doc_id:
             doc_id = f"{doc.metadata.get('original_case_id')}_{doc.metadata.get('type')}_{doc.metadata.get('p_num', '')}_{doc.metadata.get('field_name', '')}"
        if doc_id:
             all_docs_map[doc_id] = doc 

    #converting raw Chroma results back to Langchain Document format
    if filter_results and filter_results.get('ids'):
        for i in range(len(filter_results['ids'])):
            item_id = filter_results['ids'][i]
            metadata = filter_results['metadatas'][i]
            doc_text = filter_results['documents'][i] 

            if doc_text: 
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


rag_chain = (

    {"context": enhanced_retrieval_function, "input": RunnablePassthrough()}
    | qa_prompt 
    | llm 
    | StrOutputParser() 
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
            url = doc.metadata.get('original_case_id')  
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