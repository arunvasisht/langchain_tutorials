from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableBranch

from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

prompt_classification = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant."),
        ("human","please classify the review as postive, neutral or negative : {user_input}")
    ]
)

classification_chain = prompt_classification | llm | StrOutputParser()


prompt_positive = ChatPromptTemplate.from_messages(
    [
        ("system","you are a helpful assistant"),
        ("human","generate a thank you note for the feedback given by the user : {user_input} and ask him if the user would like to recommend it to others as well.")
    ]
)

prompt_neutral = ChatPromptTemplate.from_messages(
    [
        ("human","generate a note for the feedback given by the user : {user_input}. Also ask what are the areas of improvement.")
    ]
)

prompt_negative = ChatPromptTemplate.from_messages(
    [
        ("human","generate a apology message for the positive feedback given by the user : {user_input}. You should also ask the user for further details.")
    ]
)
chain_positive = prompt_positive | llm | StrOutputParser()
chain_neutral = prompt_neutral | llm | StrOutputParser()
chain_negative = prompt_negative | llm | StrOutputParser()

branches = RunnableBranch(
    (lambda x : "Positive" in x, chain_positive),
    (lambda x : "Negative" in x, chain_negative),
    chain_neutral
)


chain = classification_chain | branches

respoonse = chain.invoke({
    "user_input":"This product is great."
})

print(respoonse)


