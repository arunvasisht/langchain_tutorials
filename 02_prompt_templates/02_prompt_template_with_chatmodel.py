from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

messages = [
    ("system","You are a helpful assistant"),
    ("human","Write an poem on {topic} in {n} words")
]

prompt_template = ChatPromptTemplate.from_messages(messages=messages)

print(prompt_template.invoke(
    {
        "topic" : "G20 India",
        "n":100
    }))

print(llm.invoke(prompt_template.invoke(
    {
        "topic" : "G20 India",
        "n":100
    })))
