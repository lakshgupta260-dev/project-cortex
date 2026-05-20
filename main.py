from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core import Settings
from llama_index.core import Settings
from fastapi import FastAPI, UploadFile, File
from azure.storage.blob import BlobServiceClient
import pandas as pd
from llama_index.core import VectorStoreIndex, Document
import os

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
Settings.llm = AzureOpenAI(
    model="gpt-4.1-mini",
    deployment_name="gpt-4.1-mini",
    api_key="80aR3qn0em5AX4iq8Fu4spJVw1Oscb4ck7KqMDvZQYPH94iaoiR7JQQJ99BEAC77bzfXJ3w3AAABACOGmMi0",
    azure_endpoint="https://neoteric-ai.openai.azure.com/",
    api_version="2024-12-01-preview"
)

app = FastAPI()


CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=projectcortexlaksh123;AccountKey=znkQwBitvzLMy2qt8id2uF/ZpkkOnFRQ4U6/5nBmL/DBo1TCL0bKjpUe31Lp1h/FWCZJA8z2hAYs+ASt+8aCeA==;EndpointSuffix=core.windows.net"

index = None


@app.on_event("startup")
def startup_event():

    global index

    df = pd.read_csv("sales_data.csv")

    documents = []

    for _, row in df.iterrows():

        text = (
            f"Product: {row['product']}, "
            f"Month: {row['month']}, "
            f"Sales: {row['sales']}"
        )

        documents.append(Document(text=text))

    index = VectorStoreIndex.from_documents(documents)

    print("Index created successfully")
    
@app.get("/")
def home():

    return {
        "status": "running",
        "message": "Project Cortex FastAPI Backend"
    }


@app.post("/api/v1/files/upload")
async def upload_file(file: UploadFile = File(...)):

    blob_service_client = BlobServiceClient.from_connection_string(
        CONNECTION_STRING
    )

    container_client = blob_service_client.get_container_client("raw")

    blob_client = container_client.get_blob_client(file.filename)

    file_content = await file.read()

    blob_client.upload_blob(file_content, overwrite=True)

    return {
        "status": "success",
        "filename": file.filename
    }
@app.get("/api/v1/files/list")
def list_files():

    blob_service_client = BlobServiceClient.from_connection_string(
        CONNECTION_STRING
    )

    container_client = blob_service_client.get_container_client("raw")

    blobs = container_client.list_blobs()

    file_list = []

    for blob in blobs:

        file_list.append(blob.name)

    return {
        "files": file_list
    }
    
@app.post("/api/v1/agent/chat")
def chat(question: dict):

    user_question = question["question"]

    query_engine = index.as_query_engine()

    response = query_engine.query(user_question)

    return {
        "question": user_question,
        "answer": str(response)
    }