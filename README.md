# 🤖 Self-Correcting RAG

A powerful, production-ready **Retrieval-Augmented Generation (RAG)** system built with **LangGraph**, **Gemini**, and **Streamlit**. This project features a self-correction loop that grades document relevance and fallbacks to web search when information is missing.

---

## 🚀 Key Features

- **Self-Correction Architecture**: Uses LangGraph to grade retrieved documents and automatically trigger web search if context is insufficient.
- **Conversational Memory**: Maintains the last 5 turns of conversation for seamless follow-up questions.
- **Multi-Query Retrieval**: Automatically generates variations of your question to improve context coverage.
- **Quota Optimized**: Implements **Batch Grading** and defaults to **Gemini 3.1 Flash Lite** to save 80%+ of API quota.
- **Modern Web UI**: A beautiful Streamlit interface with live "reasoning" logs and file upload support.
- **Full Citations**: Every answer includes metadata-backed citations from your documents.

---

## 🛠️ Tech Stack

- **Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph)
- **LLM**: [Google Gemini](https://ai.google.dev/) (Flash 3.1 & 2.0)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/)
- **Web Search**: [Tavily AI](https://tavily.com/)
- **UI Framework**: [Streamlit](https://streamlit.io/)

---

## 📦 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/sathiya-22/self-correcting-rag.git
cd self-correcting-rag
```

### 2. Set up environment variables
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_gemini_key
TAVILY_API_KEY=your_tavily_key

# Optional: Override default models
# MODEL_GRADER=gemini-3.1-flash-lite-preview
# MODEL_GENERATOR=gemini-3.1-flash-lite-preview
```

### 3. Install dependencies
```bash
# Recommended: use a virtual environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Run the application
```bash
streamlit run app.py
```

---

## 📂 Project Structure

- `app.py`: Streamlit web interface.
- `graph/`: Core LangGraph logic (nodes, edges, state).
- `ingest.py`: Document processing and vector ingestion.
- `knowledge_base/`: Drop your PDFs/DOCX files here.
- `chroma_db/`: Local vector storage.

---

## 🛡️ License
MIT License. Free for personal and commercial use.
