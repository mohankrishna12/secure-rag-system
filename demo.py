import os
import sys
from dotenv import load_dotenv
from rag_pipeline import initialize_rag_pipeline
from secure_chain import create_secure_chain

# Load environment variables
load_dotenv()

def run_demo():
    print("--- Secure RAG System Demo ---")
    
    # Check for API Key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found. Please set it in a .env file or environment variables.")
        return

    # Initialize Pipeline
    print("Initializing RAG Pipeline...")
    try:
        retriever = initialize_rag_pipeline()
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        return

    # Create Chain
    print("Creating Secure Chain...")
    try:
        chain = create_secure_chain(retriever)
    except Exception as e:
        print(f"Failed to create chain: {e}")
        return

    # Test Queries
    queries = [
        "What is the total amount spent on groceries?",
        "Show me the account number for the customer.",
        "What is the current balance?",
        "List the last 3 transactions with their details.",
        "What is the phone number associated with the account?"
    ]

    print("\n--- Running Test Queries ---\n")

    for query in queries:
        print(f"User Query: {query}")
        print("Thinking...")
        try:
            response = chain.invoke(query)
            print(f"System Response: {response}\n")
            print("-" * 50)
        except Exception as e:
            print(f"Error processing query: {e}\n")

if __name__ == "__main__":
    run_demo()
