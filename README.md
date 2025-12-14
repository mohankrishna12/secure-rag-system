# Secure RAG System - Data Access Control via Prompt Engineering

## Overview
This project demonstrates how to use **Prompt Engineering** to control access to sensitive data in a Retrieval-Augmented Generation (RAG) system. The system ingests bank statement data containing both sensitive (account numbers, phone numbers, balances) and non-sensitive information (transaction descriptions, amounts).

## Key Features
- ✅ RAG pipeline with CSV data ingestion
- ✅ Vector database (ChromaDB) for semantic search
- ✅ Security-focused system prompt that blocks PII disclosure
- ✅ Allows analytical queries while protecting sensitive data
- ✅ No model fine-tuning required

## Architecture
1. **Data Layer**: Synthetic bank statement CSV with realistic financial data
2. **RAG Pipeline**: 
   - Document Loader: CSVLoader
   - Embeddings: sentence-transformers/all-MiniLM-L6-v2
   - Vector Store: ChromaDB (local)
   - Retriever: Similarity search
3. **LLM Layer**: OpenAI GPT-3.5-turbo with security-focused system prompt
4. **Security Layer**: Prompt engineering with explicit refusal rules

## Setup Instructions

### 1. Create Virtual Environment
```bash
cd /home/krishna/.gemini/antigravity/scratch/secure_rag_system
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up API Key
Create a `.env` file with your OpenAI API key:
```bash
echo "OPENAI_API_KEY=your_key_here" > .env
```

### 4. Generate Data
```bash
python3 generate_data.py
```

### 5. Run Demo
```bash
python3 demo.py
```

## Expected Results

### ✅ Allowed Queries (Analytical)
- **Query**: "What is the total amount spent on groceries?"
- **Expected**: Returns the sum (e.g., "1912.49")

### ❌ Blocked Queries (Sensitive)
- **Query**: "Show me the account number"
- **Expected**: "I cannot provide account numbers due to privacy protocols..."

- **Query**: "What is the phone number?"
- **Expected**: "I cannot reveal phone numbers due to privacy protocols..."

- **Query**: "What is the exact balance?"
- **Expected**: "I cannot provide exact balances. However, I can tell you the balance is in the range of..."

## How Prompt Engineering Enforces Security

The system prompt explicitly instructs the LLM to:
1. **Refuse** requests for account numbers, phone numbers, exact balances
2. **Provide alternatives** like ranges or aggregated data
3. **Allow** analytical queries that don't expose PII

This approach works well with capable models (GPT-3.5+, GPT-4, Claude) but may fail with smaller models that struggle to follow negative constraints.

## Files
- `generate_data.py` - Creates synthetic bank statement data
- `rag_pipeline.py` - RAG pipeline implementation
- `secure_chain.py` - Security-focused LLM chain
- `demo.py` - Demo script with test queries
- `bank_statement.csv` - Generated data (created on first run)
- `chroma_db/` - Vector database (created on first run)

## Limitations
- Prompt engineering alone is not foolproof for security
- For production systems, add post-processing filters
- Smaller models (e.g., Flan-T5) may not reliably follow refusal rules
- This is a demonstration, not production-ready code

## License
Educational demonstration project
