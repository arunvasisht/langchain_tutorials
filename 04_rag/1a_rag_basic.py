from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import os
from dotenv import load_dotenv
load_dotenv()


# paths
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir,"source","hp_1.txt")
persist_dir = os.path.join(current_dir,"db","chromadb")

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

if not os.path.exists(file_path):
    raise FileNotFoundError(f"File Does not exist at {file_path}")
else:
    if not os.path.exists(persist_dir):
        # create the vector storage and add data to it.
        
        # load text
        loader = TextLoader(file_path=file_path)
        documents = loader.load()
        #print(len(documents))

        #Split the documents
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = splitter.split_documents(documents=documents)

        for i, doc in enumerate(docs):
            doc.metadata["id"] = i+1
        
        print(f"Number of Chunks: {len(docs)}")
        print(f"Sample Chunk: {docs[1]}")

        # vectorize
        db = Chroma.from_documents(documents= docs, embedding=embeddings, persist_directory=persist_dir)

        print("vector storage created")

    else:
        print("vector storage already exists")