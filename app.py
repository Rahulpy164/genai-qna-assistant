import streamlit as st
import os
from hf_qa import query_huggingface_qa
from document_processor import (
    chunk_text, 
    preprocess_text, 
    find_relevant_chunks, 
    enhance_context_with_prompt_engineering,
    extract_key_information
)

# Set page configuration
st.set_page_config(
    page_title="GenAI Q&A Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🤖 GenAI-powered Q&A Assistant")
st.markdown("Upload documents and ask contextual questions with responses generated using open-source AI models.")

# Sidebar for document upload
with st.sidebar:
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['txt', 'pdf', 'docx'],
        help="Upload a text, PDF, or Word document"
    )
    
    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")
        
        # Read file content based on type
        if uploaded_file.type == "text/plain":
            raw_content = str(uploaded_file.read(), "utf-8")
            content = preprocess_text(raw_content)
            
            # Store processed content and chunks in session state
            st.session_state.document_content = content
            st.session_state.document_chunks = chunk_text(content)
            st.session_state.document_info = extract_key_information(content)
            
            # Display document info
            st.write("**Document Information:**")
            st.write(f"- Word count: {st.session_state.document_info['word_count']}")
            st.write(f"- Character count: {st.session_state.document_info['char_count']}")
            st.write(f"- Number of chunks: {len(st.session_state.document_chunks)}")
            
        else:
            # For now, we'll handle only text files
            # PDF and DOCX processing can be added later
            st.warning("Currently only text files are supported. PDF and DOCX support coming soon!")
            content = None
    else:
        content = None

# Main chat interface
st.header("💬 Chat Interface")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about your document..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Check if document is uploaded
    if not hasattr(st.session_state, 'document_content') or st.session_state.document_content is None:
        response = "Please upload a document first to ask questions about it."
    else:
        try:
            # Find relevant chunks for the question
            relevant_chunks = find_relevant_chunks(prompt, st.session_state.document_chunks)
            
            # Combine relevant chunks into context
            combined_context = " ".join(relevant_chunks)
            
            # Enhance context with prompt engineering
            enhanced_context = enhance_context_with_prompt_engineering(prompt, combined_context)
            
            # Query the Hugging Face model with enhanced context
            result = query_huggingface_qa(prompt, enhanced_context)
            response = result.get('answer', 'Sorry, I could not find an answer.')
            confidence = result.get('score', 0)
            
            # Add confidence score and context info to response
            response += f"\n\n*Confidence: {confidence:.2%}*"
            response += f"\n*Searched {len(relevant_chunks)} relevant text chunks*"
            
        except Exception as e:
            response = f"An error occurred while processing your question: {str(e)}"

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Information section
with st.expander("ℹ️ About this application"):
    st.markdown("""
    This Q&A Assistant uses:
    - **DistilBERT** model from Hugging Face for question answering
    - **Streamlit** for the user interface
    - **Prompt engineering** for contextual responses
    
    **Features:**
    - Upload text documents
    - Ask questions about the uploaded content
    - Get AI-powered answers with confidence scores
    - Chat-like interface for natural interaction
    
    **Future enhancements:**
    - PDF and DOCX document support
    - RAG (Retrieval-Augmented Generation) pipeline
    - Vector search capabilities
    - Multiple document support
    """)

# Clear chat button
if st.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

