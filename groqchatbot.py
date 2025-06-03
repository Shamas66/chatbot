import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# --- Load environment variables ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")


# --- Page Setup ---
st.set_page_config(page_title="📚 PDF Chatbot", page_icon="📖")
st.title("📚 PDF Chatbot")

# --- Sidebar ---
with st.sidebar:
    st.header("🛠️ Model Settings")
    temperature = st.slider("Temperature (creativity)", 0.0, 1.0, 0.7, 0.1)
    max_tokens = st.slider("Max Tokens (response length)", 256, 4096, 1024, 256)

    st.markdown("---")
    uploaded_file = st.file_uploader("📄 Upload PDF Novel", type="pdf")

    st.markdown("---")
    st.subheader("💬 Chat Summary")
    if "chat_history" in st.session_state and st.session_state.chat_history:
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            st.markdown(f"**Q{i+1}:** {chat['user']}")
    else:
        st.markdown("No questions yet.")

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# --- Load LLM with Caching ---
@st.cache_resource
def load_llm(temp, tokens):
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not found in environment. Please check your .env file.")
        st.stop()

    return ChatGroq(
        model="llama3-8b-8192",
        api_key=GROQ_API_KEY,
        temperature=temp,
        max_tokens=tokens
    )

llm = load_llm(temp=temperature, tokens=max_tokens)

# --- Load and Process PDF ---
if uploaded_file and st.session_state.vectorstore is None:
    with open("temp_novel.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp_novel.pdf")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(docs, embeddings)

    st.session_state.vectorstore = vectorstore
    st.success("✅ PDF Loaded Successfully!")

# --- Ask a Question ---
st.markdown("### 📝 Ask a Question About the Document")

query = st.chat_input("Type your question here...")

if query and st.session_state.vectorstore:
    retriever = st.session_state.vectorstore.as_retriever()

    with st.spinner("Thinking... 🤔"):
        try:
            docs = retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in docs[:5]])

            prompt = f"""
You are a helpful and knowledgeable AI Assistant. Your task is to provide a detailed and comprehensive answer to the question based on the given context.
ONLY use the CONTEXT below to answer the QUESTION. 
If the context does not contain enough information to answer the question, simply say: "I don't know." 

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""
            response = llm.invoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)

            # Save to chat history
            st.session_state.chat_history.append({
                "user": query,
                "bot": answer
            })

        except Exception as e:
            st.error(f"❌ Error: {e}")

# --- Show Chat History like ChatGPT ---
if st.session_state.chat_history:
    for chat in st.session_state.chat_history:
        st.chat_message("user").markdown(chat["user"])
        st.chat_message("assistant").markdown(chat["bot"])
