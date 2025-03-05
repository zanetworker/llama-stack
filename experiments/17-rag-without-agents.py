import os
import httpx
import asyncio
from termcolor import cprint
from typing import List, Dict, Any
from llama_stack_client import LlamaStackClient
from llama_stack_client.types import Document

class RAGWithoutAgents:
    def __init__(self, base_url: str, port: str):
        """Initialize the RAG implementation without agents"""
        self.client = LlamaStackClient(base_url=f"http://localhost:{port}")
        self.vector_db_id = "rag-without-agents-db"
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
                            document_id=f"doc-{i}",
                            content=response.text,
                            mime_type="text/plain",
                            metadata={"url": full_url, "source": "tutorial"},
                        )
                        documents.append(doc)
                        cprint(f"Document {i} loaded, length: {len(response.text)}", "green")
                    else:
                        cprint(f"Failed to fetch {full_url}: Status {response.status_code}", "red")
                except Exception as e:
                    cprint(f"Error fetching {full_url}: {str(e)}", "red")
        return documents

    def setup_vector_db(self):
        """Initialize vector database"""
        try:
            self.client.vector_dbs.delete(vector_db_id=self.vector_db_id)
            cprint(f"Deleted existing vector database: {self.vector_db_id}", "yellow")
        except Exception as e:
            cprint(f"Vector DB doesn't exist or couldn't be deleted: {str(e)}", "yellow")

        cprint("\nRegistering vector database...", "cyan")
        self.client.vector_dbs.register(
            vector_db_id=self.vector_db_id,
            embedding_model="all-MiniLM-L6-v2",
            embedding_dimension=384,
            provider_id="faiss",
        )
        cprint(f"Vector database {self.vector_db_id} registered successfully", "green")

    def rag_using_builtin_tool(self, documents: List[Document]):
        """RAG implementation using LlamaStack's built-in RAG tool"""
        cprint("\n=== RAG Using Built-in Tool ===", "yellow")

        try:
            # First, ensure documents are properly chunked before insertion
            cprint("Inserting documents into vector database...", "cyan")
            self.client.tool_runtime.rag_tool.insert(
                documents=documents,
                vector_db_id=self.vector_db_id,
                chunk_size_in_tokens=512,  # Reduced chunk size for better stability
                overlap_size_in_tokens=50   # Add overlap for context continuity
            )
            cprint("Documents inserted successfully", "green")

            # Query with more specific configuration
            query_config = {
                "max_tokens_in_context": 2048,
                "max_chunks": 3,
                "similarity_threshold": 0.7,
                "query_generator_config": {
                    "type": "default",
                    "separator": "\n"
                }
            }

            cprint("Querying vector database...", "cyan")
            results = self.client.tool_runtime.rag_tool.query(
                content=[{
                    "type": "text",
                    "text": "What are the key learning objectives across these documents?"
                }],
                vector_db_ids=[self.vector_db_id],
                query_config=query_config
            )

            cprint("\nRAG Tool Query Results:", "green")
            if hasattr(results, 'content'):
                for item in results.content:
                    if hasattr(item, 'text'):
                        print(f"\n{item.text}")
            else:
                cprint("No content found in results", "red")

        except httpx.HTTPError as e:
            cprint(f"HTTP Error in rag_tool method: {str(e)}", "red")
            if hasattr(e, 'response'):
                cprint(f"Response content: {e.response.content}", "red")
        except Exception as e:
            cprint(f"Error in rag_tool method: {str(e)}", "red")

    def rag_manual_implementation(self, documents: List[Document]):
        """RAG implementation by manually retrieving chunks and calling LLM"""
        cprint("\n=== Manual RAG Implementation ===", "yellow")

        try:
            # Make sure documents are in the vector database
            cprint("Ensuring documents are in vector database...", "cyan")
            
            # Query configuration for retrieval
            retrieval_config = {
                "max_chunks": 5,
                "similarity_threshold": 0.6,
            }
            
            # Retrieve relevant chunks for our query
            query_text = "Summarize the main concepts from these documents"
            cprint(f"Retrieving chunks for query: '{query_text}'", "cyan")
            
            chunks = self.client.vector_dbs.query(
                vector_db_id=self.vector_db_id,
                query=query_text,
                top_k=5,
                **retrieval_config
            )
            
            if not chunks or len(chunks) == 0:
                cprint("No chunks retrieved from vector database", "red")
                return
                
            cprint(f"Retrieved {len(chunks)} chunks", "green")
            
            # Prepare context from chunks
            context = "\n\n".join([chunk.text for chunk in chunks if hasattr(chunk, 'text')])
            
            # Create prompt with retrieved context
            prompt = f"""You are a helpful assistant. Use ONLY the following context to answer the question.
            
Context:
{context}

Question: {query_text}

Answer:"""
            
            # Call LLM with the prompt
            cprint("Calling LLM with retrieved context...", "cyan")
            model = os.environ.get('INFERENCE_MODEL', "Llama3.2-3B-Instruct")
            
            response = self.client.llm.generate(
                model=model,
                prompt=prompt,
                max_tokens=512,
                temperature=0.1
            )
            
            cprint("\nDirect RAG Results:", "green")
            if hasattr(response, 'text'):
                print(f"\n{response.text}")
            else:
                cprint("No text found in response", "red")
                
        except Exception as e:
            cprint(f"Error in direct_rag method: {str(e)}", "red")

    def rag_with_query_optimization(self, documents: List[Document]):
        """Advanced RAG implementation with query reformulation and chunk filtering"""
        cprint("\n=== Advanced RAG with Query Optimization ===", "yellow")
        
        try:
            # Step 1: Reformulate the query using LLM
            original_query = "What practical applications are mentioned in these documents?"
            cprint(f"Original query: '{original_query}'", "cyan")
            
            model = os.environ.get('INFERENCE_MODEL', "Llama3.2-3B-Instruct")
            reformulation_prompt = f"""You are a query optimization expert. Your task is to reformulate the following query to make it more effective for retrieval from a vector database.
            
Original query: {original_query}

Reformulated query:"""
            
            reformulation_response = self.client.llm.generate(
                model=model,
                prompt=reformulation_prompt,
                max_tokens=100,
                temperature=0.2
            )
            
            reformulated_query = reformulation_response.text.strip() if hasattr(reformulation_response, 'text') else original_query
            cprint(f"Reformulated query: '{reformulated_query}'", "green")
            
            # Step 2: Retrieve chunks with the reformulated query
            retrieval_config = {
                "max_chunks": 7,
                "similarity_threshold": 0.5,
            }
            
            chunks = self.client.vector_dbs.query(
                vector_db_id=self.vector_db_id,
                query=reformulated_query,
                top_k=7,
                **retrieval_config
            )
            
            if not chunks or len(chunks) == 0:
                cprint("No chunks retrieved from vector database", "red")
                return
                
            cprint(f"Retrieved {len(chunks)} chunks", "green")
            
            # Step 3: Filter chunks for relevance
            filtering_prompt = """You are a relevance filtering expert. For each text chunk, determine if it's relevant to the query about practical applications mentioned in documents.
            Rate each chunk on a scale of 1-10 for relevance, where 10 is highly relevant.
            
Query: What practical applications are mentioned in these documents?

"""
            
            # Add chunks to the filtering prompt
            for i, chunk in enumerate(chunks):
                if hasattr(chunk, 'text'):
                    filtering_prompt += f"\nChunk {i+1}:\n{chunk.text}\n"
            
            filtering_prompt += "\nRelevance ratings (just list the chunk numbers and their ratings, e.g., 'Chunk 1: 8'):"
            
            filtering_response = self.client.llm.generate(
                model=model,
                prompt=filtering_prompt,
                max_tokens=200,
                temperature=0.1
            )
            
            # Parse ratings (simplified parsing)
            ratings_text = filtering_response.text if hasattr(filtering_response, 'text') else ""
            cprint("Relevance ratings:", "cyan")
            print(ratings_text)
            
            # Step 4: Generate final response with filtered context
            # For simplicity, we'll use all chunks but in a real implementation
            # you would filter based on the ratings
            context = "\n\n".join([chunk.text for chunk in chunks if hasattr(chunk, 'text')])
            
            final_prompt = f"""You are a helpful assistant. Use ONLY the following context to answer the question.
            
Context:
{context}

Question: {original_query}

Answer with specific practical applications mentioned in the documents:"""
            
            final_response = self.client.llm.generate(
                model=model,
                prompt=final_prompt,
                max_tokens=512,
                temperature=0.1
            )
            
            cprint("\nAdvanced RAG Results:", "green")
            if hasattr(final_response, 'text'):
                print(f"\n{final_response.text}")
            else:
                cprint("No text found in response", "red")
                
        except Exception as e:
            cprint(f"Error in advanced_rag method: {str(e)}", "red")

async def main():
    # Initialize RAG implementation
    rag = RAGWithoutAgents(
        base_url="https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/",
        port=os.environ.get('LLAMA_STACK_PORT', '8000')
    )

    # Fetch documents
    documents = await rag.fetch_documents()
    cprint(f"\nLoaded {len(documents)} documents", "cyan")

    # Setup vector database
    rag.setup_vector_db()

    # Run the RAG methods
    rag.rag_using_builtin_tool(documents)
    rag.rag_manual_implementation(documents)
    rag.rag_with_query_optimization(documents)

if __name__ == "__main__":
    asyncio.run(main())
