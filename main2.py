from dotenv import load_dotenv

load_dotenv()

from typing import List
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


class Source(BaseModel):
    """
    Schema for a source used by the agent
    """

    url: str = Field(description="The URL of the source")


class AgentRespnse(BaseModel):
    """
    Schema  for agent response will answer and sources
    """

    answer: str = Field(description="this is agent answer to the query")
    sources: List[Source] = Field(
        default_factory=list,
        description="list of sources used by agent to generate the answer",
    )


tavily_search = TavilySearch()


llm = ChatOpenAI(model="gpt-4o")
tools = [tavily_search]

agent = create_agent(model=llm, tools=tools, response_format=AgentRespnse)


def main():
    print("Hello from langchain-endtoend 2!")
    result = agent.invoke(
        {"messages": HumanMessage(content="Job opening for ai engineer in gurugram")}
    )
    print(result)


if __name__ == "__main__":
    main()
