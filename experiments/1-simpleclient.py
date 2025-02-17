import os
from termcolor import colored

def create_http_client():
    from llama_stack_client import LlamaStackClient
    client = LlamaStackClient(
    base_url=f"http://localhost:{os.environ['LLAMA_STACK_PORT']}",
    )
    return client

def create_library_client(template="ollama"):
    from llama_stack import LlamaStackAsLibraryClient
    client = LlamaStackAsLibraryClient(template)
    client.initialize()
    return client


client = create_http_client()  # or create_http_client() depending on the environment you picked

# List available models
models = client.models.list()
print(colored("--- Available models: ---", "green"))
for m in models:
    print(f"- {m.identifier}")
print()

response = client.inference.chat_completion(
    model_id=os.environ["INFERENCE_MODEL"],
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about coding"}
    ]
)
print(response.completion_message.content)