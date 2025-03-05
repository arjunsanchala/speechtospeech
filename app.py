import streamlit as st
import wave
import tempfile
import numpy as np
import soundfile as sf
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

os.environ["OPENAI_API_KEY"] = "sk-ds-team-general-uRHEpM4v8JyZPznqvmSMT3BlbkFJPIMx3gi9v6BQOn58RbSN"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="HR Voice Assistant", page_icon="🎤", layout="wide")

@st.cache_resource
def load_vectorstore(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return FAISS.from_documents(chunks, OpenAIEmbeddings())

def record_audio(RECORD_SECONDS=5, RATE=44100, CHANNELS=1):
    status_message = st.info("🎙️ Listening... Speak now!")

    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_audio_path = temp_audio.name

    # Recording using soundfile
    audio_data = np.zeros((RATE * RECORD_SECONDS,), dtype=np.float32)
    with sf.SoundFile(temp_audio_path, mode='w', samplerate=RATE, channels=CHANNELS, format='WAV') as file:
        with sf.InputStream(samplerate=RATE, channels=CHANNELS, callback=lambda indata, frames, time, status: file.write(indata)):
            time.sleep(RECORD_SECONDS)

    status_message.empty()
    return temp_audio_path

def transcribe_audio(audio_path):
    status_message = st.info("⏳ Transcribing your voice...")
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file, response_format="text")
    status_message.empty()
    return transcript

def generate_response(query):
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
    audio_bytes = BytesIO()
    communicate = Communicate(text[:4096], "en-US-JennyNeural")
    
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_bytes.write(chunk["data"])
    
    audio_bytes.seek(0)
    return audio_bytes.getvalue()

def main():
    with st.sidebar:
        st.header("HR Policy Document")
        uploaded_file = st.file_uploader("Upload your HR policy document (TXT)", type=["txt"])
        
        if uploaded_file is not None and "vectorstore" not in st.session_state:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            st.session_state.vectorstore = load_vectorstore(tmp_path)
            st.success("Document processed successfully!")
        elif "vectorstore" in st.session_state:
            st.info("HR policy document is loaded.")

    st.title("HR Voice Assistant 🎤")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! Upload a document and ask me anything about HR policies!"}]
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "audio" in msg:
                st.audio(msg["audio"], format="audio/mp3", autoplay=True)
    
    if "waiting_for_input" not in st.session_state:
        st.session_state.waiting_for_input = True
    
    if st.session_state.waiting_for_input:
        if st.button("🎤 Start Recording"):
            st.session_state.waiting_for_input = False
            st.rerun()
    else:
        audio_path = record_audio()
        transcript = transcribe_audio(audio_path)
        os.remove(audio_path)
        
        with st.chat_message("user"):
            st.write(f"🗣️ {transcript}")
        
        response = generate_response(transcript)
        response_audio = asyncio.run(text_to_speech(response))
        
        with st.chat_message("assistant"):
            st.write(response)
            st.audio(response_audio, format="audio/mp3", autoplay=True)
        
        st.session_state.messages.append({"role": "user", "content": transcript})
        st.session_state.messages.append({"role": "assistant", "content": response, "audio": response_audio})
        
        st.session_state.waiting_for_input = True
        st.rerun()

if __name__ == "__main__":
    main()
