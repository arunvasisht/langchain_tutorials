from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel

from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

prompt_features = ChatPromptTemplate.from_messages(
    [
        ("system","You are an expert product reviewer."),
        ("human","List the main features of the product - {product_name}")
    ]
)



def pros_prompt_value(features):
    prompt_pros = ChatPromptTemplate.from_messages(
        [
            ("system","You are an expert product reviewer."),
            ("human","From the given features {given_features}, list the pros of the product.")
        ]
    )
    return prompt_pros.format_prompt(given_features = features)

def cons_prompt_value(features):
    prompt_cons = ChatPromptTemplate.from_messages(
        [
            ("system","You are an expert product reviewer."),
            ("human","From the given features {given_features}, list the cons of the product.")
        ]
    )
    return prompt_cons.format_prompt(given_features = features)


pros_branch = (
    RunnableLambda(lambda x : pros_prompt_value(x)) 
    | llm 
    | StrOutputParser()
)

cons_branch = (
    RunnableLambda(lambda x: cons_prompt_value(x))
    | llm 
    | StrOutputParser()
)

def combine_pros_cons(pros,cons):
    return f"{pros}\n\n==============================\n\n\n{cons}"


chain =(
    prompt_features
    | llm
    | StrOutputParser()
    | RunnableParallel(branches={"pros":pros_branch,"cons":cons_branch})
    | RunnableLambda(lambda x: combine_pros_cons(x["branches"]["pros"],x["branches"]["cons"]))
)

response = chain.invoke(
    {
        "product_name":"Pixel 6a Phone"
    }
)

print(response)


