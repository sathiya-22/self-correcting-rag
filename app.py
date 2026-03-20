import os
import streamlit as st
import time
from pathlib import Path
from dotenv import load_dotenv

# Load env vars
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage
from graph.graph import build_graph
from ingest import ingest, KNOWLEDGE_BASE_DIR

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Self-Correcting RAG",
    page_icon="🤖",
    layout="wide",
)

# ─── Session State ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "logs" not in st.session_state:
    st.session_state.logs = []

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ RAG Settings")
    
    st.markdown("### 📄 Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload PDFs or DOCX files", 
        type=["pdf", "docx"], 
        accept_multiple_files=True
    )
    
    if st.button("🚀 Process Documents"):
        if uploaded_files:
            with st.status("Processing documents...", expanded=True) as status:
                os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
                for uploaded_file in uploaded_files:
                    file_path = KNOWLEDGE_BASE_DIR / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.write(f"✅ Saved {uploaded_file.name}")
                
                st.write("🔄 Running ingestion pipeline...")
                ingest()
                status.update(label="✅ Ingestion Complete!", state="complete", expanded=False)
                st.success("Documents ready!")
        else:
            st.warning("Please upload files first.")

    st.divider()

    st.markdown("### 🤖 Model Selection")
    grader_model = st.selectbox(
        "Grader Model",
        ["gemini-3.1-flash-lite-preview", "gemma-3-4b-it", "gemini-2.0-flash", "gemini-flash-lite-latest"],
        index=0
    )
    generator_model = st.selectbox(
        "Generator Model",
        ["gemini-3.1-flash-lite-preview", "gemini-2.0-flash", "gemini-pro-latest"],
        index=0
    )
    
    # Update environment variables on the fly
    os.environ["MODEL_GRADER"] = grader_model
    os.environ["MODEL_GENERATOR"] = generator_model

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.logs = []
        st.rerun()

# ─── Main Chat UI ─────────────────────────────────────────────────────────────
st.title("🤖 Self-Correcting RAG Chat")
st.markdown("""
Interact with your documents using this **Self-Correcting RAG** pipeline. 
The system will search your local docs, grade them, and even search the web if needed!
""")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        log_container = st.container()
        
        with st.status("🧠 Thinking...", expanded=True) as status:
            # Prepare state
            history = [
                HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                for m in st.session_state.messages[:-1]
            ]
            
            initial_state = {
                "question": prompt,
                "context": [],
                "generation": "",
                "retry_count": 0,
                "hint": None,
                "source": "",
                "history": history,
            }
            
            # Run Graph
            graph = build_graph()
            
            # Simple way to intercept prints (hacky for streamlit but works for quick feedback)
            # In a real app, we'd use callbacks, but for this demo:
            final_state = graph.invoke(initial_state)
            
            generation = final_state.get("generation", "")
            
            if generation.startswith("__HINT_REQUIRED__:"):
                generation = generation.replace("__HINT_REQUIRED__:", "⚠️ **Information Not Found.**\n\n")
                status.update(label="🛑 Hint Required", state="error")
            else:
                status.update(label="✅ Answer Generated", state="complete")

        # Stream effect
        full_response = ""
        message_placeholder = st.empty()
        for chunk in generation.split():
            full_response += chunk + " "
            time.sleep(0.02)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
