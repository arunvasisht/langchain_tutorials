from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import os
from dotenv import load_dotenv
load_dotenv()


# paths
current_dir = os.path.dirname(os.path.abspath(__file__))
persist_dir = os.path.join(current_dir,"db","chromadb")

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

query = "Where did Harry Potter go for school?"

db = Chroma(persist_directory=persist_dir,embedding_function=embeddings)

retriever = db.as_retriever()

response  = retriever.invoke(query)

print(query)
print(response)


