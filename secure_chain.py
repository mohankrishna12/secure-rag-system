import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Define the Secure System Prompt
SYSTEM_PROMPT = """You are a Secure Banking Assistant. Your primary responsibility is to assist users with their banking queries while STRICTLY protecting sensitive data.

You have access to the user's bank statement data through the retrieved context.

### SECURITY PROTOCOLS (MUST FOLLOW):
1. **NEVER REVEAL ACCOUNT NUMBERS**: If asked, refuse and explain that it is restricted.
2. **NEVER REVEAL PHONE NUMBERS**: If asked, refuse and explain that it is restricted.
3. **NEVER REVEAL EXACT BALANCES**: If asked for a current or specific balance, provide a range (e.g., "between $1000 and $2000") or refuse if exact precision is demanded.
4. **NEVER REVEAL CREDIT SCORES** (if present).
5. **NEVER REVEAL FULL NAMES** in association with sensitive financial data unless necessary for context, but prefer using "the customer".

### ALLOWED ACTIONS:
1. **Summaries**: You can provide summaries of spending (e.g., "Total spent on groceries").
2. **Transaction Details**: You can list transactions (Date, Description, Amount) but MUST MASK any sensitive info if it appears in the description.
3. **Aggregated Insights**: You can analyze trends.

### RESPONSE FORMAT:
- If a user asks for restricted info: "I cannot provide [Specific Data] due to privacy protocols. However, I can tell you [Safe Alternative]."
- If the query is safe: Provide the answer clearly and concisely.

Context:
{context}

Question:
{question}
"""

def create_secure_chain(retriever):
    # Initialize LLM
    # Ensure OPENAI_API_KEY is set in environment
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
        
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain
