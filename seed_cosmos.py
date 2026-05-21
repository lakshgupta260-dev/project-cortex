from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, Document
from azure.cosmos import CosmosClient
import pandas as pd
import os

load_dotenv()

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

df = pd.read_csv("sales_data.csv")

documents = []

for _, row in df.iterrows():

    text = (
        f"Product: {row['product']}, "
        f"Month: {row['month']}, "
        f"Sales: {row['sales']}"
    )

    documents.append(Document(text=text))

for i, doc in enumerate(documents):

    embedding = Settings.embed_model.get_text_embedding(doc.text)

    item = {
        "id": str(i),
        "text": doc.text,
        "embedding": embedding
    }

    container.upsert_item(item)

print("Embeddings stored in Cosmos DB successfully")