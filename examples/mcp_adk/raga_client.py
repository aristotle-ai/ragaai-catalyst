import asyncio
import os
import signal
import sys
from contextlib import suppress
from dotenv import load_dotenv
import google.generativeai as genai
from google.genai import types

# Google ADK imports
from google.adk.agents import Agent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from dotenv import load_dotenv
from ragaai_catalyst import RagaAICatalyst, init_tracing
from ragaai_catalyst.tracers import Tracer
load_dotenv()


genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))


catalyst = RagaAICatalyst(
    access_key=os.getenv('RAGAAI_CATALYST_ACCESS_KEY'), 
    secret_key=os.getenv('RAGAAI_CATALYST_SECRET_KEY'), 
    base_url=os.getenv('RAGAAI_CATALYST_BASE_URL')
)

tracer = Tracer(
    project_name=os.environ['RAGAAI_PROJECT_NAME'],
    dataset_name=os.environ['RAGAAI_DATASET_NAME'],
    tracer_type="google-adk"
)

init_tracing(catalyst=catalyst, tracer=tracer)



async def run_agent_safely(runner, session_id, user_message):
    """Run the agent with proper error handling and context management"""
    try:
        response_received = False
        async for event in runner.run_async(
            user_id="user123",
            session_id=session_id,
            new_message=user_message
        ):
            if event.is_final_response():
                print(f"\n✅ Response:\n{event.content.parts[0].text}")
                response_received = True
                break
        
        if not response_received:
            print("❌ No response received from agent")
            
    except asyncio.CancelledError:
        print("❌ Agent operation was cancelled")
        raise
    except Exception as e:
        print(f"❌ Error running agent: {e}")

async def main():
    try:
        # Check required environment variable
        google_api_key = os.getenv("GOOGLE_API_KEY")

        if not google_api_key:
            print("❌ Error: GOOGLE_API_KEY environment variable is required")
            return
        # Configure API keys
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

        print("Creating File System MCP toolset...")

        # Create File System MCP toolset
        fs_toolset = MCPToolset(
            connection_params=StdioServerParameters(
                command="npx",
                args=[
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    ".",  # Use the current directory as the target
                ],
                env={}
            )
        )

        print("Creating agent...")

        # Create the agent
        agent = Agent(
            name="file_system_assistant",
            model="gemini-1.5-pro",  # Using a more capable model for complex reasoning
            instruction="""You are a helpful file system assistant that can help users manage their files and directories.
            You can help users with tasks like:
            1. Listing directory contents
            2. Creating new files and directories
            3. Reading file contents
            4. Moving and copying files
            5. Deleting files and directories
            Always confirm before performing destructive operations like delete.
            Provide clear feedback about what actions you're taking.
            If an operation fails, explain why and suggest alternatives.""",
            tools=[fs_toolset]
        )

        print("Setting up session...")

        # Set up session service
        session_service = InMemorySessionService()
        session = await session_service.create_session(
            app_name="file_system_app",
            user_id="user123",
            session_id="session123"
        )

        # Create runner
        runner = Runner(
            agent=agent,
            app_name="file_system_app",
            session_service=session_service
        )

        print("\n📁 File System Assistant is ready!")
        print("Type 'exit', 'quit', or 'q' to stop")
        print("=" * 50)

        # Interactive loop with better error handling
        while True:
            try:
                # Get user input
                user_input = input("\nHow can I help with your files? ").strip()

                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("Goodbye!")
                    break

                if not user_input:
                    continue

                print(f"\n🔍 Processing request: {user_input}")
                print("-" * 30)

                # Create message content
                user_message = types.Content(
                    role="user",
                    parts=[types.Part(text=user_input)]
                )

                # Run the agent with proper error handling
                await run_agent_safely(runner, session.id, user_message)

                print("\n" + "=" * 50)

            except KeyboardInterrupt:
                print("\n\nReceived interrupt, shutting down gracefully...")
                break
            except EOFError:
                print("\n\nEnd of input, shutting down...")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                continue

    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
    finally:
        # Perform graceful shutdown
        await graceful_shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nApplication interrupted")
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        print("Application terminated")