import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader


dotenv_path = '../.env'  # Adjust the path to where your .env file is relative to your script
load_dotenv(dotenv_path)

### Load the GROQ and OpenAI API Key

os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')

st.title("ChatGroq Demo with Llama3 Demo")

llm=ChatGroq(groq_api_key=os.getenv('GROQ_API_KEY'), 
             model_name="llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate resposne based on the question.
<context>
{context}
<context>
Question:{input}

"""
)

def vector_embedding():

    if "vectordb" not in st.session_state:

        st.session_state.embeddings=AzureOpenAIEmbeddings(azure_endpoint=os.getenv("AZURE_ENDPOINT"),
                                                        model="text-embedding-3-small",
                                                        azure_deployment="pt-sm-embd")
        st.session_state.loader=PyPDFDirectoryLoader("./us_census") ## data ingestion
        st.session_state.docs=st.session_state.loader.load()  ## document loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) ## chunk creation
        st.session_state.final_docs=st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) ## splitting
        st.session_state.vectordb=FAISS.from_documents(st.session_state.final_docs,st.session_state.embeddings) ## creating vector store with aoi embeddings


## Fields

prompt1=st.text_input("Enter your question from the documents")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector store DB is ready")

import time


if prompt1:
    document_chain=create_stuff_documents_chain(llm, prompt)
    retriever=st.session_state.vectordb.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever, document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt1})
    print("Response time :", time.process_time()-start)
    st.write(response['answer'])

    # Streamlit expander
    with st.expander("Documents similarity search"):
        # Find the relevant chunks
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("------------------------")

    



