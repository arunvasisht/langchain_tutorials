from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

load_dotenv()

#llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

prompt_input = "Write me an essay on the topic : {topic} in {n} words."

prompt_template = ChatPromptTemplate.from_template(prompt_input)

print(prompt_template.invoke({
    "topic":"cat",
    "n":100
}))