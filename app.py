import os
import streamlit as st
from dotenv import load_dotenv

from src.utils.helpers import  extract_video_id, format_time, youtube_timestamp_url
from src.loaders.youtube_loader import YouTubeTranscriptLoader
from src.processing.text_splitter import transcript_to_documents, chunk_documents
from src.vectorstore.chroma_store import build_or_load_chroma 
from src.chains.qa_chain import make_qa_chain


# App Setup

load_dotenv()
st.set_page_config(
    page_title="YouTube RAG Chatbot",
    layout="wide"
)

st.title("🎥 YouTube RAG Chatbot")

# Sidebar Controls
with st.sidebar:
    st.header("Settings")


    model_name = st.selectbox(
        "Chat model",
        ["gpt-4o-mini", "gpt-4.1-mini"],
        index=0,
    )

    chunk_size = st.slider(
        "Chunk size",
        min_value=300,
        max_value=2000,
        value=900,
        step=50,
    )

    chunk_overlap = st.slider(
        "Chunk overlap",
        min_value=0,
        max_value=400,
        value=120,
        step=10,
    )


    k = st.slider(
        "Top-K retrieved chunks",
        min_value=2,
        max_value=10,
        value=5,
        step=1,
    )

# Session State

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "video_id" not in st.session_state:
    st.session_state.video_id = None

# Build/ Load Knowledge Base
video_url = st.text_input("Paste YouTube video URL")
build_btn = st.button("Build / Load Knowledge Base")

if video_url and build_btn:
    try:
        video_id = extract_video_id(video_url)

        loader = YouTubeTranscriptLoader(languages=["en"])
        transcript_lines = loader.load(video_id)

        docs = transcript_to_documents(video_id, transcript_lines)
        chunks = chunk_documents(
            docs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap)
        
        vectorstore = build_or_load_chroma(video_id, chunks)
            
        rag_chain = make_qa_chain(
            vectorstore,
            model_name=model_name,
            k=k,
        )

        st.session_state.vectorstore = vectorstore
        st.session_state.rag_chain = rag_chain
        st.session_state.video_id = video_id

        st.success("✅ Knowledge base is ready!")
        st.caption(
            f"Video ID: {video_id} | "
            f"Transcript lines: {len(transcript_lines)} | "
            f"Chunks: {len(chunks)}"
        )
    except Exception as e:
        st.error(str(e))

# --- Ask a question ---
question = st.text_input("Ask a question about the video")
ask_btn = st.button("Ask")

if ask_btn and question:
    if st.session_state.rag_chain is None:
        st.warning("Please build the knowledge base first.")
    else:
        # 🔴 RAG QUERY (CORRECT)
        answer = st.session_state.rag_chain.invoke(question)

        st.subheader("Answer")
        st.write(answer)

    # # 🔍 DEBUG RETRIEVAL (temporary)
    #     retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": k})
    #     docs = retriever.invoke(question)

    #     st.write("🔍 Retrieved docs:", len(docs))
    #     for i, d in enumerate(docs):
    #         st.write(f"Doc {i}:", d.page_content[:200])

    #     # 🔴 RAG QUERY
    #     answer = st.session_state.rag_chain.invoke(question)

    #     st.subheader("Answer")
    #     st.write(answer)