from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableSequence
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

messages = [
    ("system","You are a helpful assitant"),
    ("human","Give me only the names of top {n} mountains in {country}")
]

prompt_template = ChatPromptTemplate.from_messages(messages)

invoke_prompt = RunnableLambda(lambda x:prompt_template.format_messages(**x))
invoke_llm = RunnableLambda(lambda x:llm.invoke(x))
invoke_output = RunnableLambda(lambda x:x.content)


chain = RunnableSequence(first=invoke_prompt,middle=[invoke_llm],last=invoke_output)

response = chain.invoke({
    "n":3,
    "country":"USA"
})

print(response)

