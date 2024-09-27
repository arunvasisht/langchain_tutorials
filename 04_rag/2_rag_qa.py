from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

import os
from dotenv import load_dotenv
load_dotenv()


# paths
current_dir = os.path.dirname(os.path.abspath(__file__))
persist_dir = os.path.join(current_dir,"db","chromadb")

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

query = "Who is Arvind Kejriwal?"

db = Chroma(persist_directory=persist_dir,embedding_function=embeddings)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

retriever = db.as_retriever()

relevant_docs  = retriever.invoke(query)


# setup the prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","you are a helpful assistant for Q/A and you need to answer user's query using only the given {context}. If you are unable to find the answer, simply say 'unable to find the answer'"),
        ("human","query : {query}")
    ]
)

context = ""
for doc in relevant_docs:
    context += doc.page_content + "\n\n"

chain = prompt | llm | StrOutputParser()

response = chain.invoke(
    {
        "context":context,
        "query":query
    }
)

print(query)
print(context)

print(response)



