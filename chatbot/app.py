from langchain_openai import AzureChatOpenAI  # Adjusted import
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

dotenv_path = '../.env'  # Adjust the path to where your .env file is relative to your script
load_dotenv(dotenv_path)

# Setting environment variables for Azure and OpenAI
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
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
st.title('Langchain Demo With Azure OpenAI API')
input_text = st.text_input("Search the topic you want")

# Adjusting the LLM initialization for Azure
llm = AzureChatOpenAI(
    model_name="gpt-4-32k",  # Specify the model name
    deployment_name="pt_rekoncile",  # Specify the deployment name if needed
    openai_api_base=os.getenv("OPENAI_API_BASE"),  # Base URL for the API
    openai_api_type=os.getenv("OPENAI_API_TYPE")   # API type (usually 'azure' or similar)
)
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({'question': input_text}))
