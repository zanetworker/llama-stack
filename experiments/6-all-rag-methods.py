import os
from termcolor import cprint
import httpx
import asyncio
from llama_stack_client import LlamaStackClient
from llama_stack_client.types import Document, AgentConfig
from llama_stack_client.lib.agents.agent import Agent
from typing import List, Dict, Any
from llama_stack_client.lib.agents.event_logger import EventLogger

class RAGImplementation:
    def __init__(self, base_url: str, port: str):
        self.client = LlamaStackClient(base_url=f"http://localhost:{port}")
        self.vector_db_id = "test-vector-db"
        self.urls = ["chat.rst", "llama3.rst", "datasets.rst", "lora_finetune.rst"]
        self.base_url = base_url

    async def fetch_documents(self) -> List[Document]:
        """Fetch documents from URLs asynchronously"""
        documents = []
        async with httpx.AsyncClient() as http_client:
            for i, url in enumerate(self.urls):
                full_url = f"{self.base_url}{url}"
                print(f"Fetching {full_url}")
                try:
                    response = await http_client.get(full_url)
                    if response.status_code == 200:
                        doc = Document(
                            document_id=f"num-{i}",
                            content=response.text,
                            mime_type="text/plain",
                            metadata={"url": full_url},
                        )
                        documents.append(doc)
                        cprint(f"Document {i} loaded, length: {len(response.text)}", "green")
                    else:
                        cprint(f"Failed to fetch {full_url}: Status {response.status_code}", "red")
                except Exception as e:
                    cprint(f"Error fetching {full_url}: {str(e)}", "red")
        return documents

    def setup_vector_db(self):
        """Initialize vector database exactly as shown in documentation"""
        try:
            self.client.vector_dbs.delete(vector_db_id=self.vector_db_id)
        except:
            pass

        cprint("\nRegistering vector database...", "cyan")
        response = self.client.vector_dbs.register(
            vector_db_id=self.vector_db_id,
            embedding_model="all-MiniLM-L6-v2",
            embedding_dimension=384,
            provider_id="faiss",
        )
        return response

    def method1_vector_io(self, documents: List[Document]):
        """Method 1: Direct Vector IO Implementation following documentation example"""
        cprint("\n=== Method 1: Vector IO ===", "yellow")

        try:
            # Convert documents to chunks exactly as shown in documentation
            chunks = []
            for doc in documents:
                chunks.append({
                    "document_id": doc.document_id,
                    "content": doc.content,
                    "mime_type": "text/plain"
                })

            # Insert chunks using keyword arguments
            self.client.vector_io.insert(
                vector_db_id=self.vector_db_id,
                chunks=chunks
            )
            cprint(f"Inserted {len(chunks)} chunks successfully", "green")

            # Query chunks exactly as shown in documentation
            chunks_response = self.client.vector_io.query(
                vector_db_id=self.vector_db_id,
                query="What are the main topics covered in these documents?",
            )

            cprint("\nVector IO Query Results:", "green")
            # Print results without any format specifiers
            for result in chunks_response:
                print("\nScore:", result[0])
                if result[1]:
                    print("Content:", result[1][:200])

        except Exception as e:
            cprint(f"Error in vector_io method: {str(e)}", "red")

    def method2_rag_tool(self, documents: List[Document]):
        """Method 2: RAG Tool Implementation following documentation example"""
        cprint("\n=== Method 2: RAG Tool ===", "yellow")

        try:
            # Insert documents using RAG Tool exactly as shown in documentation
            self.client.tool_runtime.rag_tool.insert(
                documents=documents,
                vector_db_id=self.vector_db_id,
                chunk_size_in_tokens=512,
            )

            # Query using RAG Tool
            results = self.client.tool_runtime.rag_tool.query(
                content=[{
                    "type": "text",
                    "text": "What are the key learning objectives across these documents?"
                }],
                vector_db_ids=[self.vector_db_id]
            )

            cprint("\nRAG Tool Query Results:", "green")
            if hasattr(results, 'content'):
                for item in results.content:
                    if hasattr(item, 'text'):
                        print(f"\n{item.text}")

        except Exception as e:
            cprint(f"Error in rag_tool method: {str(e)}", "red")

    def method3_agent_rag(self, documents: List[Document]):
        """Method 3: Agent-based RAG Implementation with robust error handling and logging"""
        cprint("\n=== Method 3: Agent-based RAG ===", "yellow")

        try:
            # Configure agent with optimized settings
            agent_config = AgentConfig(
                model=os.environ.get('INFERENCE_MODEL', "Llama3.2-3B-Instruct"),
                instructions="""You are a concise assistant. Use the retrieved context to answer queries. 
                              Focus on providing brief and relevant information based on the context.""",
                enable_session_persistence=False,
                max_infer_iters=2,
                toolgroups=[
                    {
                        "name": "builtin::rag",
                        "args": {
                            "vector_db_ids": [self.vector_db_id],
                            "top_k": 2,
                            "min_score": 0.7
                        }
                    }
                ]
            )

            # Create agent and session with event logging
            agent = Agent(self.client, agent_config)
            session_id = agent.create_session("rag_session")

            # Define focused prompts for better results
            user_prompts = [
                "Based on the retrieved context, list the 2 most important topics mentioned. Be very brief.",
            ]

            for prompt in user_prompts:
                cprint(f'User> {prompt}', 'green')
                try:
                    response = agent.create_turn(
                        messages=[{"role": "user", "content": prompt}],
                        session_id=session_id,
                    )

                    # Handle streaming response with event logging
                    for log in EventLogger().log(response):
                        try:
                            log.print()
                        except Exception as e:
                            cprint(f"Error printing log: {str(e)}", "red")
                            continue

                except httpx.ReadTimeout:
                    cprint("Response timed out. Try reducing the context or simplifying the query.", "red")
                except Exception as e:
                    cprint(f"Error in response handling: {str(e)}", "red")

        except Exception as e:
            cprint(f"Error in agent_rag method setup: {str(e)}", "red")

async def main():
    # Initialize RAG implementation
    rag = RAGImplementation(
        base_url="https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/",
        port=os.environ['LLAMA_STACK_PORT']
    )

    # Fetch documents
    documents = await rag.fetch_documents()
    cprint(f"\nLoaded {len(documents)} documents", "cyan")

    # Setup vector database
    rag.setup_vector_db()

    # Demonstrate all three RAG methods
    rag.method1_vector_io(documents)
    rag.method2_rag_tool(documents)
    rag.method3_agent_rag(documents)


if __name__ == "__main__":
    asyncio.run(main())