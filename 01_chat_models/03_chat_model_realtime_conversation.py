from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

chat_history = []

system_message = SystemMessage(content="You are a helpful assistant.")
chat_history.append(system_message)

while True:
    human_message = input("Human (type 'exit' to terminate the conversation): ")
    chat_history.append(HumanMessage(content=human_message))
    if human_message.lower() == 'exit':
        break
    else:
        response = llm.invoke(chat_history)
        ai_message = response.content
        print(f"AI : {ai_message}")
        chat_history.append(AIMessage(content=ai_message))


print(chat_history)
