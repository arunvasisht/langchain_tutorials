from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

response = llm.invoke("Hi! Can you write a short poem welcoming the participants in the training session")

print(response)


