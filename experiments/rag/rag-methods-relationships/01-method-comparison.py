#!/usr/bin/env python3
"""
RAG Methods Comparison Example

This script demonstrates all 3 RAG query methods in Llama Stack and shows
how they relate to each other. Each method builds upon the previous one,
adding more features and abstraction.

Run this example to see:
1. Vector IO API - Direct vector database access
2. RAG Tool API - Built on Vector IO, adds query processing
3. Agent-based RAG - Built on RAG Tool, adds conversational context
"""

import asyncio
import os
from typing import List, Dict, Any
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types import Document
from termcolor import cprint


class RAGMethodsDemo:
    def __init__(self, base_url: str = "http://localhost:8321"):
        self.client = LlamaStackClient(base_url=base_url)
        self.vector_db_id = "demo-knowledge-base"
        self.model_id = os.environ.get('INFERENCE_MODEL', "meta-llama/Llama-3.3-70B-Instruct")
        
    def setup_vector_database(self):
        """Setup vector database with sample documents"""
        cprint("\n=== Setting up Vector Database ===", "cyan")
        
        # Clean up existing database
        try:
            self.client.vector_dbs.delete(vector_db_id=self.vector_db_id)
        except:
            pass
            
        # Register new vector database
        self.client.vector_dbs.register(
            vector_db_id=self.vector_db_id,
            embedding_model="all-MiniLM-L6-v2",
            embedding_dimension=384,
            provider_id="faiss",  # or your preferred provider
        )
        
        # Sample documents about Llama Stack
        documents = [
            Document(
                document_id="doc-1",
                content="Llama Stack is a comprehensive framework for building AI applications. It provides APIs for inference, safety, memory, and tool execution.",
                mime_type="text/plain",
                metadata={"source": "overview", "topic": "framework"}
            ),
            Document(
                document_id="doc-2", 
                content="The RAG (Retrieval-Augmented Generation) system in Llama Stack supports multiple query methods including direct vector search and agent-based retrieval.",
                mime_type="text/plain",
                metadata={"source": "rag-docs", "topic": "retrieval"}
            ),
            Document(
                document_id="doc-3",
                content="Vector databases in Llama Stack support providers like Faiss, Chroma, Qdrant, and Milvus for scalable similarity search.",
                mime_type="text/plain", 
                metadata={"source": "vector-docs", "topic": "databases"}
            )
        ]
        
        # Insert documents using RAG Tool (which uses Vector IO internally)
        self.client.tool_runtime.rag_tool.insert(
            documents=documents,
            vector_db_id=self.vector_db_id,
            chunk_size_in_tokens=256
        )
        
        cprint(f"✓ Inserted {len(documents)} documents into {self.vector_db_id}", "green")

    def method_1_vector_io_api(self, query: str):
        """Method 1: Direct Vector IO API - Lowest level, maximum control"""
        cprint("\n=== Method 1: Vector IO API (Direct Database Access) ===", "yellow")
        cprint("Purpose: Direct vector database querying with minimal abstraction", "white")
        cprint("Dependencies: Vector database providers only", "white")
        
        try:
            # Direct vector database query
            response = self.client.vector_io.query_chunks(
                vector_db_id=self.vector_db_id,
                query=[{"type": "text", "text": query}],
                params={
                    "max_chunks": 3,
                    "mode": "vector",
                    "score_threshold": 0.0
                }
            )
            
            cprint(f"\n📊 Vector IO Results for: '{query}'", "green")
            cprint(f"Found {len(response.chunks)} chunks", "white")
            
            for i, (chunk, score) in enumerate(zip(response.chunks, response.scores)):
                cprint(f"\nChunk {i+1} (Score: {score:.3f}):", "blue")
                cprint(f"Content: {chunk.content}", "white")
                cprint(f"Metadata: {chunk.metadata}", "gray")
                
            return response
            
        except Exception as e:
            cprint(f"❌ Vector IO Error: {str(e)}", "red")
            return None

    def method_2_rag_tool_api(self, query: str):
        """Method 2: RAG Tool API - Built on Vector IO, adds processing"""
        cprint("\n=== Method 2: RAG Tool API (Query Processing + Context Formatting) ===", "yellow")
        cprint("Purpose: High-level RAG with built-in query processing", "white")
        cprint("Dependencies: Uses Vector IO API internally", "white")
        cprint("Relationship: CALLS → Vector IO API for chunk retrieval", "magenta")
        
        try:
            # RAG Tool query with advanced configuration
            result = self.client.tool_runtime.rag_tool.query(
                content=[{"type": "text", "text": query}],
                vector_db_ids=[self.vector_db_id],
                query_config={
                    "max_chunks": 3,
                    "max_tokens_in_context": 1024,
                    "mode": "vector",
                    "chunk_template": "Source {index}: {chunk.content}\nMetadata: {metadata}\n---\n",
                    "query_generator_config": {
                        "type": "default",
                        "separator": " "
                    }
                }
            )
            
            cprint(f"\n🔧 RAG Tool Results for: '{query}'", "green")
            cprint("Note: Context is automatically formatted and ready for LLM", "white")
            
            if result.content:
                for item in result.content:
                    if hasattr(item, 'text'):
                        cprint(f"\n{item.text}", "white")
            
            # Show metadata
            if result.metadata:
                cprint(f"\n📋 Metadata:", "blue")
                cprint(f"Document IDs: {result.metadata.get('document_ids', [])}", "gray")
                cprint(f"Scores: {result.metadata.get('scores', [])}", "gray")
                
            return result
            
        except Exception as e:
            cprint(f"❌ RAG Tool Error: {str(e)}", "red")
            return None

    def method_3_agent_based_rag(self, query: str):
        """Method 3: Agent-based RAG - Built on RAG Tool, adds conversation"""
        cprint("\n=== Method 3: Agent-based RAG (Conversational + Automatic) ===", "yellow")
        cprint("Purpose: Conversational RAG with automatic tool invocation", "white")
        cprint("Dependencies: Uses RAG Tool API via ToolRuntimeRouter", "white")
        cprint("Relationship: CALLS → ToolRuntimeRouter → RAG Tool API", "magenta")
        
        try:
            # Configure agent with RAG toolgroup
            agent_config = {
                "model": self.model_id,
                "instructions": """You are a helpful assistant with access to knowledge about Llama Stack.
                Use the retrieved context to answer questions accurately and cite your sources.""",
                "enable_session_persistence": False,
                "toolgroups": [{
                    "name": "builtin::rag",
                    "args": {
                        "vector_db_ids": [self.vector_db_id],
                        "query_config": {
                            "max_chunks": 3,
                            "max_tokens_in_context": 1024,
                            "mode": "vector"
                        }
                    }
                }]
            }
            
            # Create agent and session
            agent = Agent(self.client, agent_config)
            session_id = agent.create_session("demo_session")
            
            cprint(f"\n🤖 Agent RAG Results for: '{query}'", "green")
            cprint("Note: Agent automatically invokes RAG when needed", "white")
            
            # Agent automatically uses RAG during conversation
            response = agent.create_turn(
                messages=[{"role": "user", "content": query}],
                session_id=session_id
            )
            
            # Log the agent's workflow
            for log in EventLogger().log(response):
                log.print()
                
            return response
            
        except Exception as e:
            cprint(f"❌ Agent RAG Error: {str(e)}", "red")
            return None

    def demonstrate_relationships(self):
        """Show how the methods build upon each other"""
        cprint("\n" + "="*80, "cyan")
        cprint("RAG METHODS RELATIONSHIP DEMONSTRATION", "cyan")
        cprint("="*80, "cyan")
        
        cprint("\n🏗️  ARCHITECTURE OVERVIEW:", "yellow")
        cprint("┌─────────────────┐", "white")
        cprint("│ Agent-based RAG │ ← Highest level (conversational)", "white")
        cprint("│        ↓        │", "white")
        cprint("│  RAG Tool API   │ ← Middle level (query processing)", "white")
        cprint("│        ↓        │", "white")
        cprint("│  Vector IO API  │ ← Lowest level (direct database)", "white")
        cprint("└─────────────────┘", "white")
        
        query = "What is Llama Stack and how does it support RAG?"
        
        # Demonstrate each method
        vector_result = self.method_1_vector_io_api(query)
        rag_result = self.method_2_rag_tool_api(query)
        agent_result = self.method_3_agent_based_rag(query)
        
        # Show relationship summary
        cprint("\n" + "="*80, "cyan")
        cprint("RELATIONSHIP SUMMARY", "cyan")
        cprint("="*80, "cyan")
        
        cprint("\n🔗 Method Dependencies:", "yellow")
        cprint("• Vector IO API: Foundation - direct database access", "white")
        cprint("• RAG Tool API: Uses Vector IO + adds query processing", "white")
        cprint("• Agent-based RAG: Uses RAG Tool + adds conversation context", "white")
        
        cprint("\n📊 Feature Comparison:", "yellow")
        features = [
            ("Raw chunk retrieval", "✓", "✓", "✓"),
            ("Query processing", "✗", "✓", "✓"),
            ("Context formatting", "✗", "✓", "✓"),
            ("Conversational context", "✗", "✗", "✓"),
            ("Automatic invocation", "✗", "✗", "✓"),
            ("Session persistence", "✗", "✗", "✓"),
        ]
        
        cprint(f"{'Feature':<25} {'Vector IO':<10} {'RAG Tool':<10} {'Agent RAG':<10}", "blue")
        cprint("-" * 65, "gray")
        for feature, vec, rag, agent in features:
            cprint(f"{feature:<25} {vec:<10} {rag:<10} {agent:<10}", "white")

    def performance_comparison(self):
        """Compare performance characteristics of each method"""
        cprint("\n" + "="*80, "cyan")
        cprint("PERFORMANCE CHARACTERISTICS", "cyan")
        cprint("="*80, "cyan")
        
        cprint("\n⚡ Performance Overview:", "yellow")
        perf_data = [
            ("Method", "Latency", "Overhead", "Control", "Best For"),
            ("Vector IO", "Low", "Minimal", "High", "Custom implementations"),
            ("RAG Tool", "Medium", "Moderate", "Medium", "Standard RAG workflows"),
            ("Agent RAG", "High", "Significant", "Low", "Conversational apps"),
        ]
        
        for row in perf_data:
            if row[0] == "Method":
                cprint(f"{row[0]:<12} {row[1]:<10} {row[2]:<12} {row[3]:<10} {row[4]}", "blue")
                cprint("-" * 70, "gray")
            else:
                cprint(f"{row[0]:<12} {row[1]:<10} {row[2]:<12} {row[3]:<10} {row[4]}", "white")


def main():
    """Main demonstration function"""
    demo = RAGMethodsDemo()
    
    try:
        # Setup
        demo.setup_vector_database()
        
        # Demonstrate relationships
        demo.demonstrate_relationships()
        
        # Performance comparison
        demo.performance_comparison()
        
        cprint("\n✅ Demo completed successfully!", "green")
        cprint("\nKey Takeaways:", "yellow")
        cprint("1. Each method builds upon the previous one", "white")
        cprint("2. Higher levels add features but also overhead", "white")
        cprint("3. Choose based on your specific requirements", "white")
        cprint("4. You can migrate between methods as needs change", "white")
        
    except Exception as e:
        cprint(f"\n❌ Demo failed: {str(e)}", "red")
        cprint("Make sure Llama Stack server is running on localhost:8321", "yellow")


if __name__ == "__main__":
    main()