from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI 
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama

from dotenv import load_dotenv

dotenv_path = '../.env'  # Adjust the path to where your .env file is relative to your script
load_dotenv(dotenv_path)

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app=FastAPI(
    title="Lanchain Server",
    version="1.0",
    description="A simple API Server"
)

add_routes(
    app,
    AzureChatOpenAI(azure_endpoint=os.getenv("AZURE_ENDPOINT")),
    path="/azureopenai"
)

model=AzureChatOpenAI(
    model_name="gpt-4-32k",  # Specify the model name
    deployment_name="pt_rekoncile",  # Specify the deployment name if needed
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),  # Base URL for the API
    openai_api_type=os.getenv("OPENAI_API_TYPE"),   # API type (usually 'azure' or similar)
    openai_api_version = os.getenv("OPENAI_API_VERSION")
)

#model number 2
llm=Ollama(model="llama3")

prompt1=ChatPromptTemplate.from_template("Write me a short paragraph about {topic} with a limit of words")

prompt2=ChatPromptTemplate.from_template("Write me a short poem about {topic} which is 20 words long")

add_routes(
    app,
    prompt1|model,
    path="/essay"
)

add_routes(
    app,
    prompt2|llm,
    path="/poem"
)


if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)