import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()

if __name__ == "__main__":      
    print("Retrieving from Pinecone ....")

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

    query = "what is Pinecone in machine learning?"

    chain = PromptTemplate.from_template(template=query) | llm
    result = chain.invoke({"input": query})
    print(result.content)

    print("\nRAG generation....\n")

    vector_store = PineconeVectorStore(index_name=os.getenv("INDEX_NAME"), embedding=embeddings)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    combine_documents_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    retrieval_chain = create_retrieval_chain(vector_store.as_retriever(), combine_documents_chain)

    result = retrieval_chain.invoke({"input": query})
    print(result["answer"])