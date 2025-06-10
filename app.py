# --- ENHANCED VERSION WITH STREAMING RESPONSES ---

import streamlit as st
from openai import OpenAI
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. SETTINGS AND CONFIGURATION ---

# Set the page configuration as the first Streamlit command
st.set_page_config(page_title="Dentaly RAG Chatbot", layout="centered", page_icon="🦷")

# Configuration variables
EMBEDDINGS_FILE_PATH = 'dentaly_us_embeddings.pkl'
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"

# --- 2. INITIALIZATION AND DATA LOADING ---

# Function to initialize OpenAI client with API key
def get_openai_client():
    """Initialize OpenAI client with API key from secrets or user input."""
    api_key = None
    
    # First try to get from Streamlit secrets (for deployed version)
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except:
        pass
    
    # If no API key in secrets, ask user to input it
    if not api_key:
        st.sidebar.title("🔑 Configuration")
        api_key = st.sidebar.text_input(
            "Enter your OpenAI API Key:",
            type="password",
            help="You can get your API key from https://platform.openai.com/api-keys"
        )
        
        if not api_key:
            st.warning("⚠️ Please enter your OpenAI API key in the sidebar to use the chatbot.")
            st.info("💡 You can get your API key from [OpenAI's website](https://platform.openai.com/api-keys)")
            return None
    
    return OpenAI(api_key=api_key)

# Cached function to load the knowledge base to prevent reloading on every interaction
@st.cache_resource
def load_knowledge_base(file_path):
    """Loads the pre-computed embeddings from a pickle file."""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# Load the data
knowledge_base = load_knowledge_base(EMBEDDINGS_FILE_PATH)

# --- 3. CORE RAG (RETRIEVAL-AUGMENTED GENERATION) FUNCTIONS ---

def get_embedding(text, client):
    """Generates an embedding for a given text using OpenAI's API."""
    response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return np.array(response.data[0].embedding)

def find_top_matches(user_query_vector, kb, top_n=3):
    """Finds the most relevant documents in the knowledge base using cosine similarity."""
    if not kb or user_query_vector is None:
        return []
    
    kb_vectors = np.array([item['vector'] for item in kb])
    similarities = cosine_similarity(user_query_vector.reshape(1, -1), kb_vectors)[0]
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    
    return [kb[i] for i in top_indices]

def generate_response_stream(user_question, matched_docs, client):
    """
    Generates a streaming response from the OpenAI API, augmented with context.
    This function now returns a generator object.
    """
    context = "\n\n".join([f"Source URL: {doc['url']}\nContent: {doc['combined_text']}" for doc in matched_docs])

    system_prompt = f"""
    You are an expert assistant for the Dentaly website. Your task is to answer the user's question based *exclusively* on the provided context below.
    - Be concise and helpful.
    - If the context does not contain the answer, state that you don't have enough information from the provided sources.
    - At the end of your answer, cite the source URLs you used in a list, like this:
      Sources:
      - [Title of Page 1](URL 1)
      - [Title of Page 2](URL 2)
    """

    user_prompt = f"""
    CONTEXT:
    ---
    {context}
    ---
    USER QUESTION: {user_question}
    """
    
    # The key change: stream=True
    return client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.5,
        stream=True
    )

# --- 4. STREAMLIT USER INTERFACE ---

st.title("🦷 Dentaly AI Assistant")

# Initialize OpenAI client
client = get_openai_client()

# Check if the knowledge base was loaded successfully. If not, show an error and stop.
if knowledge_base is None:
    st.error(f"Knowledge Base file not found. Please run `create_embeddings.py` in your terminal to generate the '{EMBEDDINGS_FILE_PATH}' file.")
    st.stop()

# Only proceed if we have a valid OpenAI client
if client is None:
    st.stop()

st.markdown("This chatbot uses semantic search to answer questions based on content from the Dentaly US website.")

# Initialize chat history in session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("Ask me anything about dental care..."):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display the bot's response
    with st.chat_message("assistant"):
        with st.spinner("Searching our knowledge base..."):
            query_vector = get_embedding(prompt, client)
            top_matches = find_top_matches(query_vector, knowledge_base)

        with st.spinner("Thinking..."):
            if not top_matches:
                response_text = "I couldn't find any relevant information in our knowledge base to answer your question."
                st.markdown(response_text)
            else:
                # Use st.write_stream to display the streaming response
                stream = generate_response_stream(prompt, top_matches, client)
                response_text = st.write_stream(stream)
    
    # Add the complete assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})