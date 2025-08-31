from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client= OpenAI(
    api_key= os.getenv("GEMINI_API_KEY"),
    base_url= "https://generativelanguage.googleapis.com/v1beta/openai/"
)

text= "Hello, world! This is a test of the vector embedding process."

response= client.embeddings.create(
    model= "gemini-embedding-001",
    input= text
)

# print("response: ", response)
# print("Embedding: ", response.data[0].embedding)
print("Embedding dimensions: ", len(response.data[0].embedding))