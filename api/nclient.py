import requests
import streamlit as st

def get_openai_response(input_text):
    response=requests.post("http://localhost:8000/essay/invoke",
    json={'input':{'topic':input_text}})

    return response.json()['output']['content']

def get_ollama_response(input_text):
    response=requests.post("http://localhost:8000/poem/invoke",
    json={'input':{'topic':input_text}})

    return response.json()['output']

## streamlit framework
st.title("Langchain Demo with LLAMA3 and AzureOpenAI API")
openai_input_text=st.text_input("Write a paragraph on")
ollama_input_text=st.text_input("Write a poem on")

if openai_input_text:
    st.write(get_openai_response(openai_input_text))

if ollama_input_text:
    st.write(get_ollama_response(ollama_input_text))