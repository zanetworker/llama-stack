from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types.shared_params.url import URL
from llama_stack_client import LlamaStackClient
from termcolor import cprint
import os
import time

## Start the local MCP server
# git clone https://github.com/modelcontextprotocol/python-sdk
# Follow instructions to get the env ready
# cd examples/servers/simple-tool
# uv run mcp-simple-tool --transport sse --port 8000

model_id = os.environ["INFERENCE_MODEL"]
client = LlamaStackClient(
    base_url=f"http://localhost:{os.environ.get('LLAMA_STACK_PORT', '8080')}"
)

# Check available providers
available_providers = client.providers.list()
cprint(f"Available providers: {[p.provider_id for p in available_providers]}", "yellow")

# Find the MCP provider
mcp_provider = next((p for p in available_providers if p.provider_id == "model-context-protocol"), None)
if not mcp_provider:
    cprint("MCP provider not found. Please make sure it's installed and enabled.", "red")
    exit(1)

cprint(f"Using MCP provider: {mcp_provider.provider_id}", "green")

# Try to unregister the toolgroup if it exists
try:
    client.toolgroups.unregister(toolgroup_id="mcp::filesystem")
    cprint("Unregistered existing toolgroup", "yellow")
except Exception:
    pass  # Ignore if it doesn't exist

# Register MCP tools
try:
    client.toolgroups.register(
        toolgroup_id="mcp::filesystem",
        provider_id="model-context-protocol",
        mcp_endpoint=URL(uri="http://localhost:8000/sse"))
    cprint("Successfully registered MCP toolgroup", "green")
except Exception as e:
    cprint(f"Error registering MCP toolgroup: {e}", "red")
    exit(1)

# Define an agent with MCP toolgroup and better instructions for Llama 3.2
agent_config = AgentConfig(
    model=model_id,
    instructions="""You are a helpful assistant with access to a tool that can fetch website content.
When asked to fetch content from a URL, use the 'fetch' tool to do so.
The fetch tool requires a URL parameter.
Always use the fetch tool when asked to retrieve web content.
After fetching content, summarize it in a helpful way.""",
    toolgroups=["mcp::filesystem"],
    input_shields=[],
    output_shields=[],
    enable_session_persistence=False,
    tool_choice="auto",  # Explicitly set tool choice
    tool_prompt_format="python_list",  # Format that works well with Llama models
    max_tool_calls=3,  # Allow multiple tool calls if needed
)

# Create the agent
agent = Agent(client, agent_config)
cprint("Successfully created agent", "green")

# Create a session
session_id = agent.create_session("test-session")
cprint(f"Created session: {session_id}", "green")

# Define the prompt - make it very explicit
prompt = "Please use the fetch tool to get content from https://www.google.com and then summarize what you see."
cprint(f"User> {prompt}", "green")

try:
    # Create a turn with error handling
    response = agent.create_turn(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        session_id=session_id,
    )
    
    # Use try-except for each log to handle potential None values
    for log in EventLogger().log(response):
        try:
            if log is not None:
                log.print()
            else:
                cprint("Received None log entry", "yellow")
        except Exception as e:
            cprint(f"Error processing log: {e}", "red")
except Exception as e:
    cprint(f"Error creating turn: {e}", "red")
    import traceback
    traceback.print_exc()