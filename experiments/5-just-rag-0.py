import os
from termcolor import cprint
import httpx
import asyncio
from llama_stack_client import LlamaStackClient
from llama_stack_client.types import Document

# Initialize client
client = LlamaStackClient(base_url=f"http://localhost:{os.environ['LLAMA_STACK_PORT']}")

# Fetch document content from URLs
urls = ["chat.rst", "llama3.rst", "datasets.rst", "lora_finetune.rst"]
base_url = "https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/"


async def setup():
    documents = []
    for i, url in enumerate(urls):
        full_url = f"{base_url}{url}"
        print(f"Fetching {full_url}")
        async with httpx.AsyncClient() as http_client:
            response = await http_client.get(full_url)
            if response.status_code == 200:
                doc = Document(
                    document_id=f"num-{i}",
                    content=response.text,
                    mime_type="text/plain",
                    metadata={"url": full_url},
                )
                documents.append(doc)
                print(f"Document {i} loaded, content length: {len(response.text)}")
            else:
                print(f"Failed to fetch {full_url}: Status {response.status_code}")

    return documents


# Load documents
documents = asyncio.run(setup())
print(f"Loaded {len(documents)} documents")

# Print document info
for doc in documents:
    print(f"\nDocument {doc.document_id}:")
    print("First 200 chars:", doc.content[:200])
    print("Content length:", len(doc.content))

# Setup vector database
vector_db_id = "test-vector-db"
try:
    client.vector_dbs.delete(vector_db_id=vector_db_id)
except:
    pass

print("\nRegistering vector database...")
client.vector_dbs.register(
    vector_db_id=vector_db_id,
    embedding_model="all-MiniLM-L6-v2",
    embedding_dimension=384,
    provider_id="faiss"
)

# Insert documents
print("\nInserting documents...")
client.tool_runtime.rag_tool.insert(
    documents=documents,
    vector_db_id=vector_db_id,
    chunk_size_in_tokens=2048,
)

print("\nDocument Analysis:")
print("-" * 80)

# Query each document
for doc_id in [0, 1, 3]:
    print(f"\nDocument {doc_id}:")
    print("-" * 40)

    query = f"""Extract from document num-{doc_id}:
    1. The document title
    2. Main topics and sections
    3. Learning objectives and prerequisites
    Only include content from document {doc_id}."""

    result = client.tool_runtime.rag_tool.query(
        content=[{
            "type": "text",
            "text": query
        }],
        vector_db_ids=[vector_db_id]
    )

    if hasattr(result, 'content'):
        seen_content = set()
        for item in result.content:
            if item.type == 'text' and 'content:' in item.text:
                content = item.text.split('content:', 1)[1].strip()
                for line in content.split('\n'):
                    line = line.strip()
                    if (line and not line.startswith('..') and
                            not set(line) <= set('=-') and
                            line not in seen_content and
                            any(['*' in line,
                                 'learn' in line.lower(),
                                 'prerequisites' in line.lower(),
                                 line.isupper() or
                                 any(word in line.lower() for word in
                                     ['fine-tuning', 'meta llama', 'tutorial', 'guide'])])):
                        seen_content.add(line)
                        print(line)

    print("-" * 40)

print("-" * 80)