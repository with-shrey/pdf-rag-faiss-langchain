import os
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain import hub

load_dotenv()


if __name__ == "__main__":
    print("Starting ingestion ...")
    loader = PyPDFLoader("./react.pdf")
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(os.getenv("DB_INDEX"))


    ## Retrieval
    print("Starting retrieval ...")
    llm = ChatOpenAI()

    vector_store = FAISS.load_local(
        os.getenv("DB_INDEX"),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=vector_store.as_retriever(), combine_docs_chain=combine_docs_chain
    )
    result = retrieval_chain.invoke(input={"input": "Summarise ReAct methodology in 3 sentences"})

    print(result["answer"])