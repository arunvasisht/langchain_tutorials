from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Create Tools

@tool
def fake_weather_api(city:str) -> str:
    """return the weather conditions of a particular city

    Args:
        city (str): given city name

    Returns:
        str: weather conditions of the city
    """
    return "25 degree celcius, sunny and clear skies"

@tool
def outside_availability_checker(city:str) -> str:
    """checks the availability of outdoor sittin in a city

    Args:
        city (str): city name

    Returns:
        str: outdoor availability for a restaurant
    """
    return "outdoor seating is available"

@tool
def addition(a:int,b:int)->int:
    """Performs addition of two numbers

    Args:
        a (int): numbe one
        b (int): number two

    Returns:
        int: product of the two numbers
    """
    return a+b

tools = [fake_weather_api, outside_availability_checker, addition]


prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant."),
        ("human","{query}"),
        MessagesPlaceholder("agent_scratchpad")
    ]
)

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print(executor.invoke(
    {
        "query":"What is 4 multiplied with 4?"
    }
))
