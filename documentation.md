# 📚 PDF Chatbot – Documentation

This chatbot allows users to upload a PDF document (e.g., a novel, paper, or textbook), and ask questions about its content in a conversational manner—similar to ChatGPT. It uses **LangChain** for document processing and retrieval, **Groq's LLaMA-3** model for responses, and **Streamlit** for a user-friendly web interface.

---

## 🔍 Code Overview

The app includes the following key functionalities:

- Upload a PDF file from the sidebar.
- Extract, split, and embed text from the PDF.
- Store document chunks in a FAISS vector database.
- Use **retrieval-augmented generation (RAG)** to retrieve relevant content and formulate answers.
- Ask questions using a `chat_input` at the bottom (like ChatGPT).
- Display chat history and past queries in the sidebar and main view.

---

## ⚙️ Components Breakdown

### 1. **Streamlit Interface**

- `st.sidebar`: PDF upload, model configuration (temperature, max tokens), and chat summary.
- `st.chat_input()`: Bottom-aligned input box for questions.
- `st.chat_message()`: Used to display chat-style interactions.

---

### 2. **PDF Processing**

- **Loader**: `PyPDFLoader` loads text from the PDF.
- **Splitter**: `RecursiveCharacterTextSplitter` splits the document into manageable chunks.
- **Embeddings**: `HuggingFaceEmbeddings` generates vector representations of chunks.
- **Vector Store**: `FAISS` is used for similarity search and retrieval of relevant chunks.

---

How It Works (End-to-End Workflow)
User Uploads a PDF

File is saved locally as temp_novel.pdf.

Processed into chunks and embedded.

Vector Database (FAISS)

Chunks are stored in memory using FAISS for fast retrieval.

User Asks a Question

Input is captured via st.chat_input().

Relevant document chunks are retrieved from FAISS.

Prompt Construction

A prompt is crafted combining the question and relevant chunks.

LLM Invocation

Prompt is sent to ChatGroq model.

Response is generated and shown in a conversational format.

Chat History Storage

All Q&A pairs are stored in st.session_state.chat_history.

Displayed in both the main area (chat bubbles) and sidebar (summary list).

