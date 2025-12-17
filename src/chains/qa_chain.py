from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.documents import Document
from typing import List

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a YouTube video assistant. "
            "Answer ONLY using the provided transcript context. "
            "If the answer is not in the context, say: "
            "'I don't know based on the transcript.'"
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {input}"
        ),
    ]
)


def make_qa_chain(
        vectorstore,
        model_name: str = "gpt-4o-mini",
        k: int = 5
) -> List[Document]:
    """
    Build a modern RAG chain using: retriever → document chain → LLM    
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k":k})
    
    llm = ChatOpenAI(
        model = model_name,
        temperature=0
    )

    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt= QA_PROMPT
    )

    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=document_chain,
    )

    return rag_chain