import os
from termcolor import colored
from openai import OpenAI

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


def create_openai_client():
    # Initialize OpenAI client with API key from environment variable
    # Make sure to set OPENAI_API_KEY in your environment
    # client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    client = OpenAI(
        base_url=f"http://localhost:{os.environ['LLAMA_STACK_PORT']}/v1/openai/v1",
        api_key="dummy-key"  # Llama Stack might not check this, but the OpenAI client requires it
    )

    return client


llama_stack_client = create_http_client()

# List available models
models = llama_stack_client.models.list()
print(colored("--- Available models: ---", "green"))
for m in models:
    print(f"- {m.identifier}")
print()

response = llama_stack_client.inference.chat_completion(
    model_id=os.environ["INFERENCE_MODEL"],
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about coding"}
    ]
)


print(response.completion_message.content)

print(colored("\n--- OpenAI client ---", "green"))

open_ai_client = create_openai_client()

response = open_ai_client.chat.completions.create(
    model=os.environ["INFERENCE_MODEL"],
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about coding"}
    ],
    stream=True
)

print("Streaming response:")
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
print("\n")
# print(response.choices[0].message.content)