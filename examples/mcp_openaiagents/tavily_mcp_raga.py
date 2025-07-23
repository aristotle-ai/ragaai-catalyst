import asyncio
import os
from agents import Agent, Runner
from agents.mcp import MCPServer, MCPServerStdio
from dotenv import load_dotenv
from ragaai_catalyst import RagaAICatalyst, init_tracing
from ragaai_catalyst.tracers import Tracer
load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


catalyst = RagaAICatalyst(
    access_key=os.getenv('RAGAAI_CATALYST_ACCESS_KEY'), 
    secret_key=os.getenv('RAGAAI_CATALYST_SECRET_KEY'), 
    base_url=os.getenv('RAGAAI_CATALYST_BASE_URL')
)

tracer = Tracer(
    project_name=os.environ['RAGAAI_PROJECT_NAME'],
    dataset_name=os.environ['RAGAAI_DATASET_NAME'],
    tracer_type="agentic/openai_agents"
)

init_tracing(catalyst=catalyst, tracer=tracer)


async def run_tavily_mcp_client(mcp_server: MCPServer):
    agent = Agent(
        name="Tavily Search Assistant",
        instructions="Use the Tavily search tools to answer the user's questions with real-time web search results.",
        mcp_servers=[mcp_server],
    )
    
    while True:
        message = input("\n\nEnter your search query (or 'exit' to quit): ")
        if message.lower() == "exit" or message.lower() == "q":
            break
        
        print(f"\n\nSearching: {message}")
        result = await Runner.run(starting_agent=agent, input=message)
        print(result.final_output)

async def main():
    # Configure the Python Tavily MCP Server with tracing
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY environment variable is required")
    
    # Use the Python version of Tavily MCP server
    async with MCPServerStdio(
        name="Tavily MCP Server (Python)",
        params={
            "command": "python",
            "args": ["-m", "mcp_server_tavily"],
            "env": {
                "TAVILY_API_KEY": tavily_api_key
            }
        },
        client_session_timeout_seconds=30,
    ) as server:
        await run_tavily_mcp_client(server)

if __name__ == "__main__":
    asyncio.run(main())

#if differet tool call check for that also wether it is getting traced or not