from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

messages = [
    ("system","You are a helpful assistant"),
    ("human","Write an story on {topic} in {n} words")
]

prompt_template = ChatPromptTemplate.from_messages(messages=messages)

parser = StrOutputParser()

chain = prompt_template | llm | parser

response = chain.invoke(
    {
        "topic":"tiger",
        "n":150
    }
)

print(response)

