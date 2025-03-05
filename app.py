import os

os.environ["OPENAI_API_KEY"] = "sk-ds-team-general-uRHEpM4v8JyZPznqvmSMT3BlbkFJPIMx3gi9v6BQOn58RbSN"

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

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="HR Voice Assistant", page_icon="🎤", layout="wide")

@st.cache_resource
def load_vectorstore(file_path):
    """Load and process a document into a vector store"""
    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return FAISS.from_documents(chunks, OpenAIEmbeddings())

def simulated_record_audio():
    """
    Simulates audio recording in environments without audio input hardware.
    Instead of actual recording, it provides a text input field.
    """
    status_message = st.info("💬 Type your question below")
    
    # Create a temporary file path for compatibility with the rest of the code
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_audio_path = temp_audio.name
    
    # Instead of recording, use a text input
    user_text = st.text_input("Type your question:", key="simulated_voice_input")
    
    # Create a submit button to proceed
    if st.button("Submit", key="submit_voice_simulation"):
        # Create an empty WAV file (it won't be used for transcription)
        with wave.open(temp_audio_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b'')
        
        # Store the text input in session state to use it instead of transcription
        st.session_state.simulated_transcript = user_text
        status_message.empty()
        return temp_audio_path
    
    # If the button wasn't clicked, return None to indicate we're still waiting
    return None

def transcribe_audio(audio_path):
    """Transcribe audio or use simulated text input"""
    # If we have a simulated transcript, use it instead of actual transcription
    if hasattr(st.session_state, 'simulated_transcript'):
        transcript = st.session_state.simulated_transcript
        # Clear it for the next round
        del st.session_state.simulated_transcript
        return transcript
    
    # Otherwise proceed with normal transcription (won't be used in simulation mode)
    status_message = st.info("⏳ Transcribing your voice...")
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file, response_format="text")
    status_message.empty()
    return transcript

def generate_response(query):
    """Generate a response to the user's query based on the HR policy document"""
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
    time.sleep(1.5)  # Simulate slight delay before response
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

def main():
    """Main application function"""
    with st.sidebar:
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
    Upload an HR policy document in the sidebar, then ask your questions below.
    """)
    
    # Display Streamlit version information and autoplay note
    st.info("""
    Note: Automatic audio playback requires Streamlit version 1.10.0 or newer.
    Current installed version: """ + st.__version__ + """
    If autoplay doesn't work, you'll need to upgrade Streamlit with: `pip install --upgrade streamlit>=1.10.0`
    """)
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! Upload a document and ask me anything about HR policies!"}]
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "audio" in msg:
                # Try to use autoplay (will work on Streamlit 1.10.0+)
                try:
                    st.audio(msg["audio"], format="audio/mp3", autoplay=True)
                except:
                    st.audio(msg["audio"], format="audio/mp3")
    
    if "waiting_for_input" not in st.session_state:
        st.session_state.waiting_for_input = True
    
    if st.session_state.waiting_for_input:
        if st.button("💬 Ask a Question"):
            st.session_state.waiting_for_input = False
            st.rerun()
    else:
        # Get input from simulated audio recording
        audio_path = simulated_record_audio()
        
        # Only proceed if we got a valid input
        if audio_path is not None:
            transcript = transcribe_audio(audio_path)
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            # Don't process empty input
            if not transcript or transcript.strip() == "":
                st.warning("Please enter a question.")
                st.session_state.waiting_for_input = True
                st.rerun()
            
            with st.chat_message("user"):
                st.write(f"🗣️ {transcript}")
            
            response = generate_response(transcript)
            
            # Generate speech asynchronously
            try:
                response_audio = asyncio.run(text_to_speech(response))
            except Exception as e:
                st.warning(f"Could not generate audio response: {e}")
                response_audio = None
            
            with st.chat_message("assistant"):
                st.write(response)
                if response_audio:
                    # Try to use autoplay (will work on Streamlit 1.10.0+)
                    try:
                        st.audio(response_audio, format="audio/mp3", autoplay=True)
                    except:
                        st.audio(response_audio, format="audio/mp3")
            
            st.session_state.messages.append({"role": "user", "content": transcript})
            message_data = {"role": "assistant", "content": response}
            if response_audio:
                message_data["audio"] = response_audio
            st.session_state.messages.append(message_data)
            
            st.session_state.waiting_for_input = True
            st.rerun()

if __name__ == "__main__":
    main()