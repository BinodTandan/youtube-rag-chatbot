import os
from typing import List

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.Documents import Document


def _persist_dir_for_video(video_id: str) -> str:
    return os.path.join("data", "vectorstore", "youtube", video_id)


def build_or_load_chroma(video_id:str, docs: List[Document]) -> Chroma:
    """
    Build a persistent Chroma DB for a video_id (or load if exists).
    Stores under: data/vectorstore/youtube/<video_id>
    """
    persist_dir = _persist_dir_for_video(video_id)
    os.makedirs(persist_dir, exist_ok=True)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # If already built, load it(fast)

    existing = os.path.exists(os.path.join(persist_dir, "chorma.sqlite3")) or os.path.exists(persist_dir)

    if existing:
        vectorestore = Chroma(
            collection_name=f"yt_{video_id}",
            persist_directory=persist_dir,
            embedding_function=embeddings
        )

    # If it’s empty (first time), add docs
        if vectorestore._collection.count() == 0 and docs:
            vectorestore.add_documents(docs)
        return vectorestore
    
    # Fresh Build
    return Chroma.from_documents(
        documents= docs,
        embedding=embeddings,
        collection_name=f"yt_{video_id}",
        persist_directory=persist_dir
    )