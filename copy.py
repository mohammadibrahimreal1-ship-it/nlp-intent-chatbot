import streamlit as st
import os
import io
import hashlib
import pickle
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from typing import List

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document

# Load environment variables (API keys)
load_dotenv()

# Directory to store FAISS indexes
INDEX_DIR = "faiss_indices"
os.makedirs(INDEX_DIR, exist_ok=True)


def compute_files_hash(pdf_files) -> str:
    """Compute a sha256 hash for uploaded files to uniquely identify their content."""
    h = hashlib.sha256()
    for pdf in pdf_files:
        # read bytes (Streamlit UploadedFile)
        pdf_bytes = pdf.read()
        h.update(pdf_bytes)
        # reset file pointer so PdfReader (below) can read if needed
        pdf.seek(0)
    return h.hexdigest()


def extract_pdf_documents(pdf_docs) -> List[Document]:
    """Extract text per page from uploaded PDFs and return a list of LangChain Documents with metadata.

    Each page becomes a Document with metadata.source set to '<filename>-page-<n>'.
    """
    documents: List[Document] = []

    for pdf in pdf_docs:
        try:
            pdf_bytes = pdf.read()
            reader = PdfReader(io.BytesIO(pdf_bytes))
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    meta = {"source": f"{pdf.name}-page-{i+1}"}
                    documents.append(Document(page_content=text, metadata=meta))
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
        finally:
            # reset so Streamlit won't be broken for subsequent reads
            try:
                pdf.seek(0)
            except Exception:
                pass

    return documents


def get_text_chunks_from_documents(docs: List[Document]) -> List[Document]:
    """Split documents into chunks while preserving metadata."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunked_docs = text_splitter.split_documents(docs)
    return chunked_docs


def get_embeddings():
    """Create an OpenAIEmbeddings instance (reads OPENAI_API_KEY from env)."""
    return OpenAIEmbeddings()


def build_or_load_faiss_index(docs: List[Document], idx_hash: str):
    """Given chunked documents and a content hash, either load an existing FAISS index or build and persist a new one.

    Returns a LangChain FAISS vectorstore or None on failure.
    """
    index_path = os.path.join(INDEX_DIR, idx_hash)
    embeddings = get_embeddings()

    # If an index already exists for this doc set, load it
    if os.path.exists(index_path):
        try:
            vectorstore = FAISS.load_local(index_path, embeddings)
            return vectorstore
        except Exception as e:
            st.warning(f"Failed to load existing index, will rebuild: {e}")

    # Otherwise, build a new index and save it
    try:
        vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
        try:
            vectorstore.save_local(index_path)
        except Exception as save_err:
            st.warning(f"Index built but failed to persist to disk: {save_err}")
        return vectorstore
    except Exception as e:
        st.error(f"Failed to create FAISS vectorstore: {e}")
        return None


def get_conversation_chain(vectorstore):
    """Initialize the conversational retrieval chain with memory.

    We keep ConversationBufferMemory for short-term chat memory, but the vectorstore is persisted to disk.
    """
    llm = ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
    )
    return conversation_chain


def render_chat_history():
    """Render the stored chat history in the Streamlit chat area."""
    for msg in st.session_state.chat_history:
        role = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.write(msg["content"])
            # If assistant message contained sources, show them
            if msg.get("sources"):
                with st.expander("Sources"):
                    for src in msg["sources"]:
                        # src is a dict with 'source' and 'snippet'
                        st.markdown(f"**{src['source']}**: {src['snippet']}")


def handle_userinput(user_question: str):
    """Query the conversational chain and update history with the answer and sources."""
    if st.session_state.conversation is None:
        st.warning("Please upload and process a document first.")
        return

    try:
        # Ask the chain and request source documents
        response = st.session_state.conversation(
            {"question": user_question},
            return_source_documents=True,
        )
    except Exception as e:
        st.error(f"Error while querying the model: {e}")
        return

    # The chain typically returns 'answer' and 'source_documents'
    answer = response.get("answer") or response.get("result") or response.get("output_text")
    source_docs = response.get("source_documents") or []

    # Build a compact list of sources with small snippets for attribution
    sources_for_display = []
    for doc in source_docs:
        src = doc.metadata.get("source") if hasattr(doc, "metadata") else None
        snippet = doc.page_content[:400].strip().replace("\n", " ")
        sources_for_display.append({"source": src or "unknown", "snippet": snippet})

    # Append to session chat history
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources_for_display})

    # Re-render chat
    render_chat_history()


def main():
    st.set_page_config(page_title="Document Query Bot", page_icon="📚")

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with your PDFs 📚")

    # Main chat input
    user_question = st.chat_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    # Sidebar for upload and processing
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True,
            type=["pdf"],
        )

        if st.button("Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF.")
            else:
                with st.spinner("Processing documents..."):
                    # Compute a content-based hash to enable index reuse
                    idx_hash = compute_files_hash(pdf_docs)

                    # Try to load existing index or build a new one
                    docs = extract_pdf_documents(pdf_docs)
                    if not docs:
                        st.error("No extractable text found in the uploaded PDFs.")
                    else:
                        chunked_docs = get_text_chunks_from_documents(docs)
                        vectorstore = build_or_load_faiss_index(chunked_docs, idx_hash)
                        if vectorstore is None:
                            st.error("Failed to create or load vector store.")
                        else:
                            st.session_state.conversation = get_conversation_chain(vectorstore)
                            st.success("Ready to chat! (Index persisted and will be reused for identical uploads)")

    # Render existing chat history on load
    if st.session_state.chat_history:
        render_chat_history()


if __name__ == "__main__":
    main()
