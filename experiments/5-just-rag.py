import os
from termcolor import cprint
import httpx
import asyncio


def create_http_client():
    from llama_stack_client import LlamaStackClient
    return LlamaStackClient(base_url=f"http://localhost:{os.environ['LLAMA_STACK_PORT']}")


from llama_stack_client.types import Document

# Initialize client
client = create_http_client()


# Fetch document content from URLs
async def fetch_document_content(url):
    async with httpx.AsyncClient() as http_client:
        response = await http_client.get(url)
        if response.status_code == 200:
            return response.text
        else:
            print(f"Failed to fetch {url}: Status {response.status_code}")
    return None


# Documents to be used for RAG
urls = ["chat.rst", "llama3.rst", "datasets.rst", "lora_finetune.rst"]
base_url = "https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/"


# Create documents with actual content
async def create_documents():
    documents = []
    for i, url in enumerate(urls):
        full_url = f"{base_url}{url}"
        print(f"Fetching {full_url}")
        content = await fetch_document_content(full_url)
        if content:
            doc = Document(
                document_id=f"num-{i}",
                content=content,
                mime_type="text/plain",
                metadata={"url": full_url},
            )
            documents.append(doc)
            print(f"Document {i} loaded, content length: {len(content)}")
        else:
            print(f"Failed to load document {url}")
    return documents


def query_doc(doc_id, client, vector_db_id):
    query = f"""Analyze document num-{doc_id}. What is the title of this document 
    and what are the main topics and learning objectives listed? Only return content
    from document {doc_id}."""

    response = client.tool_runtime.rag_tool.query(
        content=[{
            "type": "text",
            "text": query
        }],
        vector_db_ids=[vector_db_id]
    )
    return response


def extract_unique_content(text):
    """Extract unique content sections from the text."""
    seen = set()
    unique_lines = []

    for line in text.split('\n'):
        line = line.strip()
        # Skip empty lines and RST directives
        if not line or line.startswith('..'):
            continue

        # Skip separator lines
        if set(line) <= set('=-'):
            continue

        if line not in seen and any([
            '*' in line,  # Bullet points
            'learn' in line.lower(),  # Learning objectives
            'prerequisites' in line.lower(),  # Prerequisites
            line.isupper() or  # Headers
            any(word in line.lower() for word in ['fine-tuning', 'meta llama', 'tutorial', 'guide'])  # Key topics
        ]):
            seen.add(line)
            unique_lines.append(line)

    return '\n'.join(unique_lines)


def main():
    # Run async setup
    documents = asyncio.run(create_documents())
    print(f"Loaded {len(documents)} documents")

    # Print document summaries
    for doc in documents:
        print(f"\nDocument {doc.document_id}:")
        print("First 200 chars:", doc.content[:200])
        print("Content length:", len(doc.content))

    # Register a vector database (clear existing first if needed)
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

    # Insert documents using the RAG tool
    print("\nInserting documents...")
    insert_response = client.tool_runtime.rag_tool.insert(
        documents=documents,
        vector_db_id=vector_db_id,
        chunk_size_in_tokens=2048,  # Larger chunks for better context
    )
    print("Insert response:", insert_response)

    print("\nDocument Analysis:")
    print("-" * 80)

    # Analyze each document
    doc_ids = [0, 1, 3]  # We know these were successfully loaded
    for doc_id in doc_ids:
        print(f"\nDocument {doc_id}:")
        print("-" * 40)

        result = query_doc(doc_id, client, vector_db_id)

        if hasattr(result, 'content'):
            doc_content = []
            for item in result.content:
                if item.type == 'text' and 'content:' in item.text:
                    content = item.text.split('content:', 1)[1].strip()
                    doc_content.append(content)

            if doc_content:
                # Get unique content from all chunks
                unique_content = extract_unique_content('\n'.join(doc_content))
                print(unique_content)

        print("-" * 40)

    print("-" * 80)


if __name__ == "__main__":
    main()