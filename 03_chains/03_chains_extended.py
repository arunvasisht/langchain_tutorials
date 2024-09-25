from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableSequence

from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

messages = [
    ("system","You are a helpful assistant"),
    ("human","Write an story on {topic} in {n} words")
]

prompt_template = ChatPromptTemplate.from_messages(messages)

def do_uppercase(story):
    print(story)
    return story.upper()


def count_words(story):
    print(story)
    return len(story.split())

lambda_uppercase = RunnableLambda(lambda x: do_uppercase(x))

lambda_wordcount = RunnableLambda(lambda x : f"Word Count: {count_words(x)}")

chain = (
    prompt_template
    | llm
    | StrOutputParser()
    | lambda_uppercase
    | lambda_wordcount
)

response = chain.invoke(
    {
        "topic":"India",
        "n":150
    }
)

print(response)

