from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core import Settings
from fastapi import FastAPI, UploadFile, File
from azure.storage.blob import BlobServiceClient
from fastapi.middleware.cors import CORSMiddleware
from azure.cosmos import CosmosClient
from fastapi import Request
from fastapi.responses import PlainTextResponse
from fastapi import Query
import pandas as pd
import requests
import os

load_dotenv()

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
CONNECTION_STRING = os.getenv("CONNECTION_STRING")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("COSMOS_KEY")
COSMOS_DATABASE = os.getenv("COSMOS_DATABASE")
COSMOS_CONTAINER = os.getenv("COSMOS_CONTAINER")

cosmos_client = CosmosClient(
    COSMOS_ENDPOINT,
    COSMOS_KEY
)

database = cosmos_client.get_database_client(
    COSMOS_DATABASE
)

container = database.get_container_client(
    COSMOS_CONTAINER
)

embed_model = HuggingFaceEmbedding(
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def chunk_text(text, chunk_size=500):

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):

        chunk = " ".join(
            words[i:i + chunk_size]
        )

        chunks.append(chunk)

    return chunks


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

    container_client = blob_service_client.get_container_client(
        "raw"
    )

    blob_client = container_client.get_blob_client(
        file.filename
    )

    file_content = await file.read()

    blob_client.upload_blob(
        file_content,
        overwrite=True
    )

    extracted_texts = []

    if file.filename.endswith(".csv"):

        temp_file_path = file.filename

        with open(temp_file_path, "wb") as f:

            f.write(file_content)

        df = pd.read_csv(temp_file_path)

        for index, row in df.iterrows():

            text = " | ".join(
                [f"{col}: {row[col]}" for col in df.columns]
            )

            extracted_texts.append(text)

    elif file.filename.endswith(".txt"):

        text = file_content.decode("utf-8")

        chunks = chunk_text(text)

        extracted_texts.extend(chunks)

    elif file.filename.endswith(".pdf"):

        from pypdf import PdfReader

        temp_file_path = file.filename

        with open(temp_file_path, "wb") as f:

            f.write(file_content)

        reader = PdfReader(temp_file_path)

        for page in reader.pages:

            text = page.extract_text()

            if text:

                chunks = chunk_text(text)

                extracted_texts.extend(chunks)

    elif file.filename.endswith(".docx"):

        from docx import Document

        temp_file_path = file.filename

        with open(temp_file_path, "wb") as f:

            f.write(file_content)

        doc = Document(temp_file_path)

        full_text = []

        for para in doc.paragraphs:

            full_text.append(para.text)

        doc_text = "\n".join(full_text)

        chunks = chunk_text(doc_text)

        extracted_texts.extend(chunks)

    for index, text in enumerate(extracted_texts):

        embedding = embed_model.get_text_embedding(
            text
        )

        item = {
            "id": f"{file.filename}-{index}",
            "file_name": file.filename,
            "text": text,
            "embedding": embedding
        }

        container.upsert_item(item)

    return {
        "status": "success",
        "filename": file.filename
    }


@app.get("/api/v1/files/list")
def list_files():

    blob_service_client = BlobServiceClient.from_connection_string(
        CONNECTION_STRING
    )

    container_client = blob_service_client.get_container_client(
        "raw"
    )

    blobs = container_client.list_blobs()

    file_list = []

    for blob in blobs:

        file_list.append(blob.name)

    return {
        "files": file_list
    }


@app.delete("/api/v1/files/delete/{file_name}")
def delete_file(file_name: str):

    try:

        blob_service_client = BlobServiceClient.from_connection_string(
            CONNECTION_STRING
        )

        container_client = blob_service_client.get_container_client(
            "raw"
        )

        blob_client = container_client.get_blob_client(
            file_name
        )

        blob_client.delete_blob()

        query = f"SELECT * FROM c WHERE c.file_name = '{file_name}'"

        items = list(
            container.query_items(
                query=query,
                enable_cross_partition_query=True
            )
        )

        for item in items:

            container.delete_item(
                item=item["id"],
                partition_key=item["id"]
            )

        return {
            "status": "success",
            "message": f"{file_name} deleted successfully"
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }


@app.post("/api/v1/agent/chat")
def chat(question: dict):

    user_question = question["question"]

    query_embedding = embed_model.get_text_embedding(
        user_question
    )

    query = """
    SELECT TOP 5 c.text, c.file_name
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

    sources = []

    for item in results:

        file_name = item.get(
            "file_name",
            "Unknown File"
        )

        if file_name not in sources:

            sources.append(file_name)

        if len(sources) >= 3:

            break

    prompt = f"""
    You are a data analyst AI.
    Prioritize the most relevant source.
    Ignore unrelated context.
    Use only context directly related to the question.
    Answer ONLY using the provided context.

Context:
{context}

Question:
{user_question}
"""

    response = Settings.llm.complete(prompt)

    return {
        "question": user_question,
        "answer": str(response),
        "sources": sources
    }


@app.get("/webhook")
async def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge")
):

    print("VERIFY TOKEN FROM META:", hub_verify_token)
    print("VERIFY TOKEN FROM ENV:", VERIFY_TOKEN)

    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:

        print("WEBHOOK VERIFIED SUCCESSFULLY")

        return PlainTextResponse(
            content=hub_challenge,
            status_code=200
        )

    print("WEBHOOK VERIFICATION FAILED")

    return PlainTextResponse(
        content="Verification failed",
        status_code=403
    )


@app.post("/webhook")
async def whatsapp_webhook(request: Request):

    try:

        payload = await request.json()

        print("Webhook Payload:", payload)

        entry = payload.get("entry", [])

        if not entry:

            print("NO ENTRY FOUND")

            return {"status": "no entry"}

        changes = entry[0].get("changes", [])

        if not changes:

            print("NO CHANGES FOUND")

            return {"status": "no changes"}

        value = changes[0].get("value", {})

        messages = value.get("messages")

        if not messages:

            print("NO MESSAGES FOUND")

            return {"status": "no messages"}

        message = messages[0]

        if "text" not in message:

            print("NON TEXT MESSAGE")

            return {"status": "non text"}

        sender = message.get("from")

        user_question = message.get(
            "text",
            {}
        ).get(
            "body",
            ""
        )

        print("User Message:", user_question)

        query_embedding = embed_model.get_text_embedding(
            user_question
        )

        query = """
        SELECT TOP 5 c.text, c.file_name
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
You are a helpful AI assistant.

Answer the user's question using the provided context.

Context:
{context}

Question:
{user_question}
"""

        response = Settings.llm.complete(prompt)

        answer = str(response)

        print("AI Response:", answer)

        url = f"https://graph.facebook.com/v22.0/{PHONE_NUMBER_ID}/messages"

        headers = {
            "Authorization": f"Bearer {WHATSAPP_TOKEN}",
            "Content-Type": "application/json"
        }

        data = {
            "messaging_product": "whatsapp",
            "to": sender,
            "text": {
                "body": answer
            }
        }

        whatsapp_response = requests.post(
            url,
            headers=headers,
            json=data
        )

        print(
            "WhatsApp API Response:",
            whatsapp_response.status_code,
            whatsapp_response.text
        )

        return {
            "status": "success"
        }

    except Exception as e:

        print("Webhook Error:", str(e))

        return {
            "error": str(e)
        }