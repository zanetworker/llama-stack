from llama_stack.distribution.library_client import LlamaStackAsLibraryClient
import asyncio

async def init_client():
    client = LlamaStackAsLibraryClient(
        "ollama",
        # provider_data is optional, but if you need to pass in any provider specific data, you can do so here.
        # provider_data = {"tavily_search_api_key": os.environ['TAVILY_SEARCH_API_KEY']}
    )
    await client.initialize()
    return client

# Run the async initialization
client = asyncio.run(init_client())
