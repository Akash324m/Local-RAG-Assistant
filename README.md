# 🧠 Local RAG Assistant: Private Document Q&A

The Local RAG Assistant is a powerful, secure, and entirely local application for querying custom documents. Built using Streamlit for the user interface and LangChain for the Retrieval-Augmented Generation (RAG) pipeline, this project allows users to ask natural language questions and receive accurate answers grounded in their private document set. It leverages Ollama to run both the Language Model (LLM) and the embedding model locally, ensuring data privacy and an offline-capable architecture.

## ✨ Key Features

*   **100% Local Execution:** All core components, including the LLM and the vector database, run locally via Ollama and Chroma, guaranteeing data stays on the user's machine.
*   **Retrieval-Augmented Generation (RAG):** Uses a sophisticated RAG pipeline to retrieve relevant text chunks from custom documents, providing the LLM with context for generating highly accurate and factual answers.
*   **Modern Language Models:** Utilizes the Phi-3 Mini LLM for high-quality, concise answer generation and the All-MiniLM model for efficient document chunk embedding.
*   **Persisted Knowledge Base:** The document knowledge base is built and persisted using Chroma, allowing for fast startup and reuse without re-processing documents.
*   **Intuitive UI with Streamlit:** Features a user-friendly web interface for document interaction, displaying response time metrics and allowing users to view the exact source context used by the model for transparency.
*   **Scalable Document Handling:** Capable of loading and processing documents from a specified directory (.txt files supported) to build a custom, searchable knowledge base.
*   **Advanced Retrievability:** Incorporates smart retrieval mechanisms (e.g., top k=2 similarity search) to fetch the most relevant content, ensuring context quality for the LLM.

## ⚙️ Technical Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| RAG Orchestration | **LangChain** | Pipelines for retrieval, prompt templating, and model chaining. |
| Large Language Model (LLM) | **Ollama (Phi-3 Mini)** | Local hosting and inference of the generative model. |
| Embedding Model | **Ollama (All-MiniLM)** | Converts text chunks into vector embeddings for the database. |
| Vector Database | **Chroma** | Stores and retrieves vector embeddings (knowledge base). |
| Frontend/UI | **Streamlit** | Creates the interactive, fast, and responsive web application. |
| Data Loading | **DirectoryLoader** | Automatically loads and processes documents from a local path. |

## 🚀 Getting Started

### Prerequisites

*   **Python 3.8+**
*   **Ollama:** Must be installed and running on your machine.
*   Pull the required models:
    ```bash
    ollama pull phi3:mini
    ollama pull all-minilm
    ```
*   **Local Document Setup:** Create a directory (e.g., `documents`) and place your `.txt` files inside.

### Installation

1.  Clone the repository:
    ```bash
    git clone [your-repo-link]
    cd local-rag-assistant
    ```
2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  Ensure Ollama is running in the background.
2.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
3.  The application will open in your web browser. The system will automatically build the vector store on the first run.
