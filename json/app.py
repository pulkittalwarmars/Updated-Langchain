import argparse
import os
import sys
# sys.path.append(os.getcwd())

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate,HumanMessagePromptTemplate
from langchain_openai import AzureChatOpenAI 
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence

from dotenv import load_dotenv
from prompt import TEST_PROMPT_TEMPLATE,NER_PROMPT_TEMPLATE

import json
import pandas as pd


dotenv_path = '../.env'  # Adjust the path to where your .env file is relative to your script
load_dotenv(dotenv_path)

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

model=AzureChatOpenAI(
    model_name="gpt-4-32k",  # Specify the model name
    deployment_name="pt_rekoncile",  # Specify the deployment name if needed
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),  # Base URL for the API
    openai_api_type=os.getenv("OPENAI_API_TYPE"),   # API type (usually 'azure' or similar)
    openai_api_version = os.getenv("OPENAI_API_VERSION")
)

# template=TEST_PROMPT_TEMPLATE
template=NER_PROMPT_TEMPLATE

prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("""You are an AI assistant that helps people find information."""),
    HumanMessagePromptTemplate.from_template(template)
])

chain = prompt_template | model | StrOutputParser()


df = pd.read_csv('./datasets/sample-test.csv')

for desc in df['GDM_ProductDesc']:
    try:
        result=chain.invoke(input={'product_desc': desc})
        print(result)
    except Exception as e:
        print(f"Error processing description '{desc}': {str(e)}")

# ner_annotation = []
# for desc in df['GDM_ProductDesc']:
#     desc = str(desc).strip()
#     try:
#         output = llm_chain.invoke(
#                     product_desc=desc
#                 )
#         output = eval(output.strip())

#         entities = []
#         for dict_ in output:
#             value = dict_["value"]

#             start = desc.find(value)
#             if start != -1:
#                 end = start + len(value)
#                 entities.append(
#                     [start, end, dict_["entity"],dict_["value"]]
#                 )

#         ner_annotation.append(
#             {
#                 "entities": entities,
#                 "text": desc
#             }
#         )
#     except Exception as e:
#             print(f"Error processing description '{desc}': {str(e)}")

# file_name = args.input_csv.split("/")[-1].replace(".csv", "_llm_ner.json")
# save_path = os.path.join(args.output_path, file_name)
# with open(save_path, "w", encoding="utf-8") as fp:
#     json.dump(ner_annotation, fp, indent=3)

