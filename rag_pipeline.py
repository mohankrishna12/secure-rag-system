import os
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuration
DATA_FILE = "bank_statement.csv"
PERSIST_DIRECTORY = "chroma_db"

def initialize_rag_pipeline():
    """
    Ingests data, creates embeddings, and initializes the vector store.
    Returns the retriever.
    """
    print("Loading data...")
    loader = CSVLoader(file_path=DATA_FILE)
    documents = loader.load()
    
    print(f"Loaded {len(documents)} documents.")
    
    # Since CSV rows are small, we might not need heavy splitting, 
    # but it's good practice to ensure chunks fit context.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    splits = text_splitter.split_documents(documents)
    
    print("Initializing embeddings (this may take a moment)...")
    # Using a local lightweight model to avoid API costs for embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("Creating/Loading Vector Store...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return retriever

if __name__ == "__main__":
    initialize_rag_pipeline()
    print("RAG Pipeline initialized successfully.")
