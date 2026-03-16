from dotenv import load_dotenv
from pathlib import Path

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

# Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIR = Path(__file__).parent / "vector_db"
CHUNK_SIZE = 1000


# Load URLs
def load_documents(urls):
    loader = UnstructuredURLLoader(urls=urls)
    return loader.load()


# Split documents
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=100
    )
    return splitter.split_documents(docs)


# Create vector store
def create_vector_store(docs):

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(VECTOR_DIR)
    )

    return vector_store


# Process URLs pipeline
def process_urls(urls):

    documents = load_documents(urls)

    chunks = split_documents(documents)

    vector_store = create_vector_store(chunks)

    return vector_store


# Ask question
def ask_question(vector_store, question):

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=500
    )

    docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
Answer the question based only on the context below.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    return response.content


if __name__ == "__main__":

    urls = [
        "https://www.opendoor.com/articles/understanding-fundamentals-of-real-estate-market",
        "https://www.realpha.com/blog/real-estate-analysis-key-considerations"
    ]

    vector_store = process_urls(urls)

    question = "What factors influence real estate markets?"

    answer = ask_question(vector_store, question)

    print(answer)