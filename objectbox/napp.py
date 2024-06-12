import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_objectbox.vectorstores import ObjectBox

load_dotenv()

## load the groq and openai API Keys
os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')
groq_api_key=os.getenv('GROQ_API_KEY')


## Streamlit 

st.title("Objectbox VectorstoreDB with Llama3 Demo")

llm=ChatGroq(
    api_key=groq_api_key,
    model="llama3-8b-8192"
)

## Prompt
prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.

    <context>
    {context}
    <context>
    Questions:{input} 

    """
)

## Vector embedding and storing in objectbox VS db

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            model="text-embedding-3-small",
            azure_deployment="pt-sm-embd"
            )
        st.session_state.loader=PyPDFDirectoryLoader("./us_census") ## Data ingestion
        st.session_state.docs=st.session_state.loader.load() ## we are loading all our documents
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        st.session_state.vectors=ObjectBox.from_documents(st.session_state.final_documents, st.session_state.embeddings, embedding_dimensions=768)

## Create stuff document chain for context
input_prompt=st.text_input("Enter your question from the documents:")

if st.button("Create embeddings DB"):
    vector_embedding()
    st.write("Objectbox DB is ready")

import time

if input_prompt:
    document_chain=create_stuff_documents_chain(llm, prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrival_chain=create_retrieval_chain(retriever, document_chain)
    start=time.process_time()

    response=retrival_chain.invoke({"input":input_prompt})

    print("Response time:", time.process_time()-start)
    st.write(response['answer'])

    ## Lets also look at the meta data

    with st.expander("Documents similarity search"):
        # Find the relevant chunks
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("-----------------------")











