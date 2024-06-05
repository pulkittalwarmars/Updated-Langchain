from langchain_openai import AzureChatOpenAI  # Adjusted import
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st

import os
from dotenv import load_dotenv

dotenv_path = '../.env'  # Adjust the path to where your .env file is relative to your script
load_dotenv(dotenv_path)

# Setting environment variables for Azure and OpenAI
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question:{question}")
    ]
)

## Streamlit Framework
st.title('Langchain Demo With Ollama llama3')
input_text = st.text_input("Search the topic you want")

# Adjusting the LLM initialization for Ollama
llm = Ollama(model="llama3")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({'question': input_text}))
