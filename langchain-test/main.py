import os
import sys

from dotenv import load_dotenv
from tools import calculator
from langchain_groq import ChatGroq
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate
from tools import calculator

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"]
)

tools = [calculator]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_openai_tools_agent(
    llm,
    tools,
    prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

question = sys.argv[1]

response = agent_executor.invoke({
    "input": question
})

print(response["output"])
