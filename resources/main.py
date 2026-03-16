import streamlit as st
from rag import process_urls, ask_question

st.title("RAG Research Tool")

# Sidebar inputs
url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

process_url = st.sidebar.button("Process URLs")

# Store vector DB in session
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

placeholder = st.empty()

if process_url:
    urls = [url for url in (url1, url2, url3) if url != ""]

    if len(urls) == 0:
        placeholder.text("You must provide at least one URL")
    else:
        placeholder.text("Processing URLs...")
        vector_store = process_urls(urls)
        st.session_state.vector_store = vector_store
        placeholder.text("Processing complete. Ask a question below.")

# Question input
query = st.text_input("Question")

if query and st.session_state.vector_store:
    answer = ask_question(st.session_state.vector_store, query)
    st.header("Answer:")
    st.write(answer)