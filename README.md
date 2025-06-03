# 📚 PDF Chatbot with LangChain, Groq, and Streamlit

An interactive PDF chatbot built with [Streamlit](https://streamlit.io), powered by the **LLaMA-3 model via Groq API**, and using **LangChain** for RAG (retrieval-augmented generation).

## ✨ Features

- 📄 Upload a PDF (e.g., a novel, research paper, manual).
- 🤖 Ask questions using a ChatGPT-style interface.
- 🔍 Answers are grounded in your document using RAG (via FAISS).
- 🎛️ Customize the response creativity (`temperature`) and length (`max tokens`) from the sidebar.
- 📜 View chat history and past questions in the sidebar.
- 🧠 Built with LLaMA-3 (via Groq) + HuggingFace sentence embeddings.

---

## 🛠️ Requirements

- Python 3.9+
- [Groq API Key](https://console.groq.com/keys)
- pip (Python package manager)

---

## 📦 Installation

1. **Clone the repository**:

```bash
git clone https://github.com/your-username/pdf-chatbot-groq.git
cd pdf-chatbot-groq
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app

```bash
streamlit run groqchatbot.py
```