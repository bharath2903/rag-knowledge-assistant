#  Real Estate Assistant — RAG-Powered Q&A with LLaMA 3 & LangChain

> **Generative AI · Retrieval-Augmented Generation · Vector Databases · LLM Integration**  
> An AI assistant that answers real estate queries by retrieving context directly from user-provided URLs — grounded in sources, not guesswork.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [How It Works](#-how-it-works)
- [Technical Architecture](#-technical-architecture)
- [Key Features](#-key-features)
- [Performance](#-performance)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Usage](#-usage)
- [Tech Stack](#-tech-stack)
- [Skills Demonstrated](#-skills-demonstrated)

---

## Overview

This project builds an **AI-powered real estate research assistant** using Retrieval-Augmented Generation (RAG). Instead of relying solely on a language model's training data, the assistant dynamically ingests content from user-provided URLs — news articles, mortgage reports, market analyses — and answers questions grounded in that retrieved context.

The system uses **LLaMA 3 via Groq API** for fast, low-latency inference, **HuggingFace embeddings** for semantic search, and **ChromaDB** as a persistent vector store. The entire pipeline is wrapped in an interactive **Streamlit** frontend.

> This approach reduces LLM API costs by ~70% compared to naive full-context prompting, while improving factual accuracy through source-backed responses.

---

##  Demo

Users enter up to 3 real estate URLs, click **"Process URLs"**, and the assistant ingests, chunks, embeds, and indexes the content. They can then ask any question and receive a context-aware answer with source citations.

!<img width="1426" height="743" alt="image" src="https://github.com/user-attachments/assets/0ae7a052-fc25-4b54-ac96-e9547a6989b5" />


---

##  How It Works

```
User provides URLs
       ↓
UnstructuredURLLoader (LangChain)
extracts raw text from web pages
       ↓
Text Splitter chunks content
into overlapping segments
       ↓
HuggingFace Embeddings
convert chunks to vectors
       ↓
ChromaDB stores vectors
in a persistent local index
       ↓
User asks a question
       ↓
Semantic similarity search
retrieves top-k relevant chunks
       ↓
LLaMA 3 (via Groq API) generates
a context-aware answer with sources
```

---

##  Technical Architecture

### Document Processing
- **`UnstructuredURLLoader`** (LangChain) — extracts and cleans raw text from any URL
- **`RecursiveCharacterTextSplitter`** — chunks text with configurable size and overlap to preserve context across boundaries

### Embeddings
- **HuggingFace `all-MiniLM-L6-v2`** — lightweight, fast sentence embedding model for semantic similarity

### Vector Store
- **ChromaDB** — local persistent vector database for storing and retrieving document embeddings via cosine similarity search

### LLM
- **LLaMA 3** served via **Groq API** — ultra-low latency inference (~250ms) without GPU infrastructure

### RAG Chain
- **`RetrievalQA` chain** (LangChain) — ties together retrieval and generation with source attribution

### Frontend
- **Streamlit** — interactive web UI for URL input, query submission, and answer display

---

##  Key Features

- **Dynamic URL ingestion** — paste any real estate URL and the system processes it on the fly
- **Semantic search** — questions are matched to the most relevant chunks using vector similarity, not keyword matching
- **Context-aware responses** — the LLM only sees retrieved context, not the entire document, keeping prompts lean and costs low
- **Source citation** — every answer includes the URLs it drew from, enabling fact verification
- **Persistent vector store** — ChromaDB persists the index between sessions so URLs don't need to be re-processed
- **URL validation & exception handling** — gracefully handles broken links and unparseable pages

---

##  Performance

| Metric | Result |
|--------|--------|
| LLM API cost reduction vs. naive full-context | ~70% |
| Reduction in manual research time | ~50% |
| Factual grounding | Significantly improved via retrieval-based context injection |
| Inference latency (Groq) | ~250ms per query |

> RAG avoids sending entire documents to the LLM on every query. Only the top retrieved chunks are included in the prompt — dramatically cutting token usage and cost.

---

## Project Structure

```
real-estate-rag-assistant/
│
├── resources/
│   ├── main.py              # Streamlit app — UI and user interaction
│   ├── rag.py               # RAG pipeline — loading, chunking, embedding & retrieval
│   ├── requirements.txt     # Python dependencies
│   └── .env                 # API keys (not committed — see setup)
│
└── README.md

```

---

##  Setup & Installation

**Prerequisites:** Python 3.10+

1. **Clone the repository**
   ```bash
   git clone https://github.com/bharath2903/real-estate-rag-assistant.git
   cd real-estate-rag-assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API key**

   Create a `.env` file in the root directory:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

   Get your free Groq API key at [console.groq.com](https://console.groq.com).

---

##  Usage

```bash
streamlit run main.py
```

Then open your browser at `http://localhost:8501` and:

1. Paste 1–3 real estate article URLs in the sidebar
2. Click **"Process URLs"** — the system ingests and indexes the content
3. Type your question in the main input field
4. Receive a sourced, context-grounded answer instantly

**Example questions you can ask:**
- *"What are the current mortgage rate trends?"*
- *"What cities have the highest housing demand right now?"*
- *"How does the Fed rate affect real estate prices?"*

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1.0+-000000?style=flat)
![Groq](https://img.shields.io/badge/Groq-LLaMA3-F55036?style=flat)
![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorStore-orange?style=flat)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-FFD21F?style=flat&logo=huggingface&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-1.55-FF4B4B?style=flat&logo=streamlit&logoColor=white)

| Category | Tool |
|----------|------|
| LLM | LLaMA 3 via Groq API |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | ChromaDB |
| RAG Framework | LangChain (`RetrievalQA`, `UnstructuredURLLoader`) |
| Frontend | Streamlit |
| Environment | python-dotenv |

---

##  Skills learnt

- **Retrieval-Augmented Generation (RAG)** — end-to-end pipeline from URL ingestion to sourced LLM response
- **Vector Databases** — embedding, indexing, and similarity search with ChromaDB
- **LLM Integration** — connecting LLaMA 3 via Groq API for low-latency inference
- **LangChain** — chaining loaders, splitters, retrievers, and LLMs into a coherent pipeline
- **Prompt Engineering** — structuring context injection to maximize factual accuracy and minimize token usage
- **Embeddings** — semantic search using HuggingFace sentence transformers
- **Streamlit Deployment** — interactive, user-facing ML app with real-time query handling

---

##  License

This project was completed as part of a  data science bootcamp at CodeBasics. Feel free to use it as a reference or build upon it.

---

<p align="center">Built with 🦙 LLaMA 3 · ⚡ Groq · 🔗 LangChain</p>
