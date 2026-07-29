import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load environment variables (API keys)
load_dotenv()

def get_pdf_text(pdf_docs):
    """Extracts text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            # Check if text is extracted successfully before adding
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
    return text

def get_text_chunks(text):
    """Splits the extracted text into manageable chunks for the AI."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200, # Overlap prevents cutting off important context
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    """Creates a FAISS vector database from text chunks."""
    # Using OpenAI embeddings to convert text into numbers
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    """Initializes the chatbot chain with memory."""
    llm = ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo")
    
    # Memory allows the bot to remember previous questions in the session
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    # Creates a chain that links the LLM, the vectorstore (knowledge), and memory
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    """Processes user input, queries the chain, and updates the chat UI."""
    # If the chain hasn't been created yet (no document uploaded), do nothing
    if st.session_state.conversation is None:
        st.warning("Please upload and process a document first.")
        return

    # Pass the user question to the conversational chain
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Display the chat history in the Streamlit interface
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            # User message
            with st.chat_message("user"):
                st.write(message.content)
        else:
            # Bot message
            with st.chat_message("assistant"):
                st.write(message.content)

def main():
    """Main Streamlit application layout and logic."""
    st.set_page_config(page_title="Document Query Bot", page_icon="📚")

    # Initialize session state variables if they don't exist
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with your PDFs 📚")
    
    # The main chat input area
    user_question = st.chat_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    # Sidebar for uploading and processing documents
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        
        if st.button("Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF.")
            else:
                with st.spinner("Processing documents..."):
                    # 1. Get pdf text
                    raw_text = get_pdf_text(pdf_docs)
                    
                    if not raw_text:
                        st.error("Could not extract text from the uploaded PDFs.")
                    else:
                        # 2. Get the text chunks
                        text_chunks = get_text_chunks(raw_text)
                        
                        # 3. Create vector store
                        vectorstore = get_vectorstore(text_chunks)
                        
                        # 4. Create conversation chain
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        st.success("Ready to chat!")

if __name__ == '__main__':
    main()
