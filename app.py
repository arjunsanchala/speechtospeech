import streamlit as st
import wave
import tempfile
import numpy as np
from io import BytesIO
import asyncio
from edge_tts import Communicate
from openai import OpenAI
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
import time
import streamlit.components.v1 as components
import base64
import re

# Hardcode OpenAI API Key
os.environ["OPENAI_API_KEY"] = "sk-ds-team-general-uRHEpM4v8JyZPznqvmSMT3BlbkFJPIMx3gi9v6BQOn58RbSN"
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

st.set_page_config(page_title="Voice Assistant", page_icon="🎤", layout="wide")

@st.cache_resource
def load_vectorstore(file_path, file_type):
    """Load and process a document (TXT or PDF) into a vector store"""
    if file_type == "txt":
        loader = TextLoader(file_path)
    elif file_type == "pdf":
        loader = PyMuPDFLoader(file_path)
    else:
        st.error("Unsupported file format.")
        return None
    
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return FAISS.from_documents(chunks, OpenAIEmbeddings())

def save_audio_file(uploaded_file):
    """Save uploaded audio file to a temporary WAV file"""
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_audio_path = temp_audio.name
    
    # Write audio data to the temporary file
    with open(temp_audio_path, 'wb') as f:
        f.write(uploaded_file.getvalue())
    
    return temp_audio_path

def transcribe_audio(audio_path):
    """Transcribe audio using OpenAI's Whisper model"""
    status_message = st.info("⏳ Transcribing your voice...")
    
    try:
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file, 
                response_format="text"
            )
        status_message.empty()
        return transcript
    except Exception as e:
        status_message.empty()
        st.error(f"Error transcribing audio: {str(e)}")
        return None

def generate_response(query):
    """Generate a response to the user's query based on the document"""
    if "vectorstore" not in st.session_state:
        return "Please upload an a document first."
    
    template = """Answer strictly based on the provided document:
    {context}
    
    Question: {question}
    
    If the answer isn't found in the document, respond: "I am sorry but this information isn't available in the document."
    """
    
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4", temperature=0),
        chain_type="stuff",
        retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )
    
    status_message = st.info("⚡ Generating response...")
    response = qa_chain.invoke({"query": query})["result"]
    status_message.empty()
    
    almost_there_message = st.info("✅ Almost there...")
    time.sleep(0.5)  # Simulate slight delay before response
    almost_there_message.empty()
    
    return response

def detect_language(text):
    """Detect if the text is in English"""
    english_pattern = re.compile(r'[A-Za-z]')  # Checks for English letters
    return "en" if english_pattern.search(text) else "ar"

async def text_to_speech(text):
    lang = detect_language(text)
    voice = "ar-SA-ZariyahNeural" if lang == "ar" else "en-US-JennyNeural"

    audio_bytes = BytesIO()
    communicate = Communicate(text[:4096], voice)
    
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_bytes.write(chunk["data"])
    
    audio_bytes.seek(0)
    return audio_bytes.getvalue()

def process_query_and_generate_response(query):
    response = generate_response(query)
    
    try:
        response_audio = asyncio.run(text_to_speech(response))
    except Exception as e:
        st.warning(f"Could not generate audio response: {e}")
        response_audio = None
    
    st.session_state.messages.append({"role": "user", "content": query})
    message_data = {"role": "assistant", "content": response}
    if response_audio:
        message_data["audio"] = response_audio
    st.session_state.messages.append(message_data)
    
    st.session_state.waiting_for_response = False
    st.session_state.show_another_question_button = True
    st.session_state.new_question = False
    
    st.rerun()

def main():
    """Main application function"""
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! Upload a document and ask me anything from the document."}]
    
    if "waiting_for_response" not in st.session_state:
        st.session_state.waiting_for_response = False
        
    if "new_question" not in st.session_state:
        st.session_state.new_question = True
        
    if "show_another_question_button" not in st.session_state:
        st.session_state.show_another_question_button = False
        
    with st.sidebar:
        st.header("Document Upload Section")
        uploaded_file = st.file_uploader("Upload your document (TXT or PDF)", type=["txt", "pdf"])
        
        if uploaded_file is not None and "vectorstore" not in st.session_state:
            file_type = uploaded_file.name.split(".")[-1]
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            with st.spinner("Processing document..."):
                st.session_state.vectorstore = load_vectorstore(tmp_path, file_type)
            st.success("Document processed successfully!")
        elif "vectorstore" in st.session_state:
            st.info("The document is loaded.")

    st.title("Voice Assistant 🎤")
    st.markdown("""
    This assistant helps answer questions from the uploaded documents.
    Upload a document in the sidebar, then ask your questions using your voice or text.
    """)
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "audio" in msg:
                st.audio(msg["audio"], format="audio/mp3", autoplay=True)
    
    if st.session_state.show_another_question_button:
        if st.button("Ask Another Question", type="primary", use_container_width=True):
            st.session_state.show_another_question_button = False
            st.session_state.new_question = True
            st.rerun()
    elif st.session_state.new_question and not st.session_state.waiting_for_response:
        st.subheader("Ask Your Question")
        
        voice_tab, text_tab = st.tabs(["Voice Input", "Text Input"])
        
        with voice_tab:
            st.write("Click the microphone button and speak your question:")
            audio_bytes = st.audio_input(label="Record your question")
            
            if audio_bytes:
                st.session_state.waiting_for_response = True
                
                audio_path = save_audio_file(audio_bytes)
                
                transcript = transcribe_audio(audio_path)
                
                if transcript:
                    process_query_and_generate_response(transcript)
        
        with text_tab:
            user_text = st.text_input("Type your question:", key="text_input")
            
            if st.button("Submit Question", key="submit_text_question"):
                if user_text and user_text.strip():
                    st.session_state.waiting_for_response = True
                    process_query_and_generate_response(user_text)
                else:
                    st.error("Please enter a question before submitting.")

if __name__ == "__main__":
    main()
