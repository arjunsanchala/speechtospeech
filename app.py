import streamlit as st
import wave
import tempfile
import numpy as np
from io import BytesIO
import asyncio
from edge_tts import Communicate
from openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
import time
import streamlit.components.v1 as components
import base64

# Initialize OpenAI client
client = None  # Will be initialized with user's API key

st.set_page_config(page_title="HR Voice Assistant", page_icon="🎤", layout="wide")

@st.cache_resource
def load_vectorstore(file_path):
    """Load and process a document into a vector store"""
    loader = TextLoader(file_path, encoding='utf-8')
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
    global client
    
    if client is None:
        st.error("Please enter your OpenAI API key in the sidebar.")
        return None
    
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
    """Generate a response to the user's query based on the HR policy document"""
    global client
    
    if client is None:
        return "Please enter your OpenAI API key in the sidebar."
    
    if "vectorstore" not in st.session_state:
        return "Please upload an HR policy document first."
    
    template = """Answer strictly based on the provided HR policy document:
    {context}
    
    Question: {question}
    
    If the answer isn't found in the document, respond: "This information isn't available in our policies. Please contact HR directly."
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

async def text_to_speech(text):
    """Convert text to speech using edge_tts"""
    audio_bytes = BytesIO()
    communicate = Communicate(text[:4096], "en-US-JennyNeural")
    
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_bytes.write(chunk["data"])
    
    audio_bytes.seek(0)
    return audio_bytes.getvalue()

def process_query_and_generate_response(query):
    # Generate text response
    response = generate_response(query)
    
    # Generate speech asynchronously
    try:
        response_audio = asyncio.run(text_to_speech(response))
    except Exception as e:
        st.warning(f"Could not generate audio response: {e}")
        response_audio = None
    
    # Update chat history
    st.session_state.messages.append({"role": "user", "content": query})
    message_data = {"role": "assistant", "content": response}
    if response_audio:
        message_data["audio"] = response_audio
    st.session_state.messages.append(message_data)
    
    # Set state to show "Ask another question" button
    st.session_state.waiting_for_response = False
    st.session_state.show_another_question_button = True
    st.session_state.new_question = False
    
    # Force rerun to update UI
    st.rerun()

def main():
    """Main application function"""
    global client
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! Upload a document and ask me anything about HR policies!"}]
    
    if "waiting_for_response" not in st.session_state:
        st.session_state.waiting_for_response = False
        
    if "new_question" not in st.session_state:
        st.session_state.new_question = True
        
    if "show_another_question_button" not in st.session_state:
        st.session_state.show_another_question_button = False
        
    with st.sidebar:
        st.header("Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Enter your OpenAI API Key:",
            type="password",
            help="Your API key is required for voice transcription and generating responses.",
            key="api_key"
        )
        os.environ["OPENAI_API_KEY"] = api_key
        
        # Initialize OpenAI client when API key is provided
        if api_key:
            client = OpenAI(api_key=api_key)
            st.success("API key set! ✅")
        else:
            st.warning("Please enter your OpenAI API key to use this application.")
        
        st.header("HR Policy Document")
        uploaded_file = st.file_uploader("Upload your HR policy document (TXT)", type=["txt"])
        
        if uploaded_file is not None and "vectorstore" not in st.session_state:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            with st.spinner("Processing document..."):
                st.session_state.vectorstore = load_vectorstore(tmp_path)
            st.success("Document processed successfully!")
        elif "vectorstore" in st.session_state:
            st.info("HR policy document is loaded.")

    st.title("HR Voice Assistant 🎤")
    st.markdown("""
    This assistant helps answer questions about HR policies based on uploaded documents.
    Upload an HR policy document in the sidebar, then ask your questions using your voice or text.
    """)
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "audio" in msg:
                st.audio(msg["audio"], format="audio/mp3", autoplay=True)
    
    # Show "Ask another question" button after response or show input widgets
    if st.session_state.show_another_question_button:
        if st.button("Ask Another Question", type="primary", use_container_width=True):
            st.session_state.show_another_question_button = False
            st.session_state.new_question = True
            st.rerun()
    elif st.session_state.new_question and not st.session_state.waiting_for_response:
        # Main UI for asking questions
        st.subheader("Ask Your Question")
        
        # Create tabs for voice and text input
        voice_tab, text_tab = st.tabs(["Voice Input", "Text Input"])
        
        # Show voice input only if API key is set
        with voice_tab:
            if client is None:
                st.warning("Please enter your OpenAI API key in the sidebar to use voice input.")
            else:
                st.write("Click the microphone button and speak your question:")
                audio_bytes = st.audio_input(label="Record your question")
                
                if audio_bytes:
                    st.session_state.waiting_for_response = True
                    
                    # Save the audio bytes to a temporary file
                    audio_path = save_audio_file(audio_bytes)
                    
                    # Transcribe the audio
                    transcript = transcribe_audio(audio_path)
                    
                    if transcript:
                        # Process the query and generate response
                        process_query_and_generate_response(transcript)
        
        with text_tab:
            user_text = st.text_input("Type your question:", key="text_input")
            
            if st.button("Submit Question", key="submit_text_question"):
                if user_text and user_text.strip():
                    st.session_state.waiting_for_response = True
                    
                    # Process the query and generate response
                    process_query_and_generate_response(user_text)
                else:
                    st.error("Please enter a question before submitting.")

if __name__ == "__main__":
    main()