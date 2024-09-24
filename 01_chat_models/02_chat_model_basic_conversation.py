from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

messages = [
    SystemMessage(content="You are a helpful assistant who is good at solving mathematics problems."),
    HumanMessage(content="What is 4 + 4"),
    AIMessage(content="8"),
    HumanMessage(content="8/8?")
]

response = llm.invoke(messages)

print(response.content)