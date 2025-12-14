import os
import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

class SecureRAG:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        
        print("Loading local model (google/flan-t5-base)... this may take a minute...")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            temperature=0,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        self.llm = HuggingFacePipeline(pipeline=pipe)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = None
        self.retriever = None
        self.chain = None
        
        self._initialize_pipeline()
        
    def _initialize_pipeline(self):
        # 1. Load Data
        loader = CSVLoader(file_path=self.csv_path)
        documents = loader.load()
        
        # 2. Create Vector Store
        self.vectorstore = Chroma.from_documents(
            documents=documents, 
            embedding=self.embeddings,
            collection_name="bank_data"
        )
        
        # 3. Setup Retriever
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # 4. Define Prompt
        # T5 works best with a clear instruction format.
        system_template = """You are a Secure Banking Assistant. Use the context below to answer the question.
        
        Context:
        {context}
        
        SECURITY RULES:
        1. Mask Account IDs (e.g., 100XXXXX).
        2. Mask Phone Numbers (e.g., 555-XXXX).
        3. Do NOT reveal exact balances. Give ranges or summaries.
        4. Do NOT reveal credit scores. Say "Good" or "Excellent".
        5. If asked for sensitive info, refuse politely.
        
        Question: {question}
        
        Answer:"""
        
        prompt = ChatPromptTemplate.from_template(system_template)
        
        # 5. Build Chain
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
    def ask(self, query):
        return self.chain.invoke(query)

if __name__ == "__main__":
    # Test run
    try:
        rag = SecureRAG("bank_statement.csv")
        print(rag.ask("What is John Doe's account number?"))
    except Exception as e:
        print(f"Error: {e}")
