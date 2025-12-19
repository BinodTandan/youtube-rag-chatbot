from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import Any

def make_qa_chain(
        vectorstore,
        model_name: str = "gpt-4o-mini",
        k: int = 5
) -> Any:
    """
    Explicit RAG pipeline:
    query → retriever → context → prompt → LLM → string
    """ 
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k":k, "lambda_mult": 0.5})
    
    model = ChatOpenAI(
        model = model_name,
        temperature=0,
    )

    parser = StrOutputParser()

    # Prompt Template
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a helpful assistant.\n"
            "Answer ONLY using the transcript context.\n"
            "If the context is insufficient, say \"I don't know.\".\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n"
            "Answer:"
        ),
    )

    #Document -> content string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    parallel_chain = RunnableParallel(
        {"context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
        }
    )

    final_chain = parallel_chain | prompt | model | parser

    return final_chain



