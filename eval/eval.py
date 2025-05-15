import json
import os

import time
from dotenv import load_dotenv
from typing import List
from datasets import Dataset
import asyncio # Import asyncio for asynchronous sleeping
from typing import List, Dict, Any # Import Any here
from bert_score import score

from ragas.evaluation import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
    ResponseGroundedness,
)
from ragas.llms import LangchainLLMWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from query import rag_chain

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 1. Instantiate your Gemini LLM via LangChain
llm_client = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)
RATE_LIMIT_RPM = 15

# 2. Wrap it for RAGas evaluation
wrapped_llm = LangchainLLMWrapper(langchain_llm=llm_client)

# Load evaluation data from JSON file
def load_eval_data(file_path: str) -> List[dict]:
    with open(file_path, "r") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise TypeError(
            f"Expected a JSON list in {file_path}, but got {type(raw).__name__}."
        )

    eval_data = []
    for item in raw:
        question = item.get("question", "")
        contexts = [c.strip() for c in item.get("contexts", [])]
        ground_truth = item.get("ground_truth", "")
        answer = item.get("answer", "")  
        
        # or generate via your RAG chain

        # try:
        #     answer = rag_chain.invoke(question)
        # except Exception as e:
        #     print(f"Error invoking RAG chain for question '{question}': {e}")
        #     answer = "Error generating answer." 

        eval_data.append({
            "question": question,
            "contexts": contexts,
            "answer": answer,
            "ground_truth": ground_truth,
        })
    return eval_data

embeddings_client = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

# Evaluate function
def evaluate_rag_system(data: List[dict]) -> Any:
    """Evaluates the RAG system using RAGas metrics."""
    ds = Dataset.from_list(data)
    print("Evaluating RAG system...")
    start = time.time()

    # Direct (synchronous) call to evaluate()
    results = evaluate(
        ds,
        metrics=[
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
            ResponseGroundedness(),
        ],
        llm=wrapped_llm,
        embeddings=embeddings_client,
    )

    elapsed = time.time() - start
    print(f"Evaluation completed in {elapsed:.2f}s")
    return results

if __name__ == "__main__":
    data_file = "evaluation_data.json"
    print(f"Loading evaluation data from {data_file}...")
    try:
        data = load_eval_data(data_file)
        if not data:
            print(f"No evaluation data loaded from {data_file}. Please check the file content.")
            exit(1)

        results = evaluate_rag_system(data)

        print("\nEvaluation Results:")

        candidates = [item["answer"] for item in data]
        references = [item["ground_truth"] for item in data]

        # Compute BERTScore
        P, R, F1 = score(candidates, references, lang="en", verbose=True)

        # Display average BERTScore F1
        print(f"Average BERTScore F1: {F1.mean().item():.4f}")
        # Convert to dict for easy printing
        try:
            results_dict = results.to_dict()
        except AttributeError:
            results_dict = results.to_pandas().to_dict("records")[0]

        for metric, score in results_dict.items():
            if isinstance(score, (int, float)):
                print(f"{metric}: {score:.4f}")
            else:
                print(f"{metric}: {score}")

    except FileNotFoundError:
        print(f"Error: Evaluation data file not found at {data_file}.")
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")

