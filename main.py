from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core import Settings
from llama_index.core import Settings
from fastapi import FastAPI, UploadFile, File
from azure.storage.blob import BlobServiceClient
import pandas as pd
from azure.cosmos import CosmosClient
from llama_index.core import VectorStoreIndex, Document
import os

load_dotenv()
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
CONNECTION_STRING = os.getenv("CONNECTION_STRING")

COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("COSMOS_KEY")
COSMOS_DATABASE = os.getenv("COSMOS_DATABASE")
COSMOS_CONTAINER = os.getenv("COSMOS_CONTAINER")
cosmos_client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)

database = cosmos_client.get_database_client(COSMOS_DATABASE)

container = database.get_container_client(COSMOS_CONTAINER)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
Settings.llm = AzureOpenAI(
    model="gpt-4.1-mini",
    deployment_name="gpt-4.1-mini",
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2024-12-01-preview"
)

app = FastAPI()


index = None


@app.on_event("startup")
async def startup_event():

    print("Application started")

@app.get("/")
def home():

    return {
        "status": "running",
        "message": "Project Cortex v3"
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

    query_embedding = Settings.embed_model.get_text_embedding(
        user_question
    )

    query = """
    SELECT TOP 3 c.text
    FROM c
    ORDER BY VectorDistance(c.embedding, @embedding)
    """

    parameters = [
        {
            "name": "@embedding",
            "value": query_embedding
        }
    ]

    results = list(
        container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        )
    )

    context = "\n".join(
        [item["text"] for item in results]
    )

    prompt = f"""
    Context:
    {context}

    Question:
    {user_question}
    """

    response = Settings.llm.complete(prompt)

    return {
        "question": user_question,
        "answer": str(response)
    }