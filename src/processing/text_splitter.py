from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.helpers import extract_video_id
from src.loaders.youtube_loader import YouTubeTranscriptLoader


def transcript_to_documents(video_id: str, lines: list) -> List[Document]:
        """Convert transcript lines -> LangChain Documents with timestamp metadata."""
        docs: List[Document] = []
        for line in lines:
                docs.append(
                        Document(
                                page_content=line.text,
                                metadata={'video_id': video_id, 'start': line.start, 'duration': line.duration}
                        )
                )
        return docs

def chunk_documents(
                docs: List[Document],
                chunk_size: int = 900,
                chunk_overlap: int = 120
) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
        )
        return splitter.split_documents(docs)



