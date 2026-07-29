# nlp-intent-chatbot

# 📚 Enterprise RAG Document Chatbot

An end-to-end Retrieval-Augmented Generation (RAG) application that allows users to seamlessly "chat" with their PDF documents. The system extracts text, vectorizes the semantic meaning, and generates context-aware responses with exact source citations.

Developed by Adeena as an interactive machine learning and NLP project.

## 🛠️ Technology Stack
* **Framework:** Streamlit (for the interactive chat UI)
* **Orchestration:** LangChain
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Embeddings & LLM:** OpenAI (`gpt-3.5-turbo` & `text-embedding-ada-002`)
* **Document Processing:** PyPDF2

## ⚙️ Core Architecture & Features
1. **Intelligent Chunking:** Implements LangChain's `CharacterTextSplitter` with defined overlaps to maintain contextual boundaries across large documents.
2. **Optimized Vector Persistence:** Calculates content hashes for uploaded PDFs and saves the FAISS index locally (`faiss_indices/`). This prevents redundant, costly API calls for identical documents by loading the persistent index from disk.
3. **Conversational Memory:** Utilizes `ConversationBufferMemory` to maintain chat history state, allowing for human-like, multi-turn follow-up questions.
4. **Source Attribution:** Responses are bundled with a "Sources" expander, displaying the exact filenames and textual snippets used by the LLM to formulate its answer, mitigating hallucinations.

## 🚀 Run Locally
```bash
# Clone the repository
git clone [https://github.com/your-username/nlp-intent-chatbot.git](https://github.com/your-username/nlp-intent-chatbot.git)
cd nlp-intent-chatbot

# Install pinned dependencies
pip install -r requirements.txt

# Set up environment variables
# Create a .env file and add: OPENAI_API_KEY=your_api_key_here

# Launch the application
streamlit run copy.py
