from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

client= OpenAI(
    base_url= "https://generativelanguage.googleapis.com/v1beta/openai/"
)


#vector embedding
embedding_model= GoogleGenerativeAIEmbeddings(
    model= "models/gemini-embedding-001",
    google_api_key= os.getenv("OPENAI_API_KEY")
)



#connect to vector db
vector_db= QdrantVectorStore.from_existing_collection(
    url= "http://localhost:6333",
    collection_name= "Learning-VectorDB",
    embedding= embedding_model
)


#take user query
query= input("👤: ")


#vector similarity search [query] in vector db. automatically create embedding of [query] using [embedding_model]
search_result= vector_db.similarity_search(
    query= query
)


#print("search_result: ", search_result)


context= "\n\n\n".join([f"Page Content: {result.page_content}\nPage Number: {result.metadata.get("page")}\nFile Location: {result.metadata.get("source")}" for result in search_result])


SYSTEM_PROMPT= f"""
    You are a helpful AI assistant.You answer user query
    based on available context retrieved from a PDF document
    along with page_content and source page number.

    You should only answer based on the context provided
    and help user get page number for reference.

    Context: {context}
"""


#print("SYSTEM_PROMPT: ", SYSTEM_PROMPT)



response= client.chat.completions.create(
    model= "gemini-2.5-pro",
    messages= [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query}
    ]
)


print(f"🤖: {response.choices[0].message.content}")