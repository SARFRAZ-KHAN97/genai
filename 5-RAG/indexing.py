from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()



pdf_path= Path(__file__).parent / "aws-overview.pdf"

#loading
loader= PyPDFLoader(file_path= pdf_path)
docs= loader.load()            # read pdf file and split into pages

#print("docs[0]: ", docs[3])


#chunking
text_splitter= RecursiveCharacterTextSplitter(
    chunk_size= 1000,
    chunk_overlap= 200
)

split_docs= text_splitter.split_documents(documents= docs)


#embedding
# embedding_model= OpenAIEmbeddings(
#     model= "gemini-embedding-001"
# )
embedding_model= GoogleGenerativeAIEmbeddings(
    model= "models/gemini-embedding-001",
    google_api_key= os.getenv("OPENAI_API_KEY")
)


#using [embedding_model] to create embeddings of [split_docs] and store them in vector db

vector_store= QdrantVectorStore.from_documents(
    documents= split_docs,
    url= "http://localhost:6333",
    collection_name= "Learning-VectorDB",
    embedding= embedding_model
)



print("Indexing of documents is complete!")