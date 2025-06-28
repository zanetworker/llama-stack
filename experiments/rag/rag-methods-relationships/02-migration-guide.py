#!/usr/bin/env python3
"""
RAG Methods Migration Guide

This script demonstrates how to migrate between different RAG query methods
in Llama Stack, showing the relationships and how to convert from one approach
to another based on changing requirements.

Migration Paths:
1. Vector IO → RAG Tool API (Add query processing)
2. RAG Tool API → Agent-based RAG (Add conversational context)
3. Agent-based RAG → RAG Tool API (Remove agent overhead)
4. RAG Tool API → Vector IO (Maximum performance)
"""

import os
from typing import List, Dict, Any, Optional
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.types import Document
from termcolor import cprint


class RAGMigrationGuide:
    def __init__(self, base_url: str = "http://localhost:8321"):
        self.client = LlamaStackClient(base_url=base_url)
        self.vector_db_id = "migration-demo-kb"
        self.model_id = os.environ.get('INFERENCE_MODEL', "meta-llama/Llama-3.3-70B-Instruct")
        
    def setup_demo_data(self):
        """Setup vector database for migration examples"""
        cprint("\n=== Setting up Demo Data ===", "cyan")
        
        try:
            self.client.vector_dbs.delete(vector_db_id=self.vector_db_id)
        except:
            pass
            
        self.client.vector_dbs.register(
            vector_db_id=self.vector_db_id,
            embedding_model="all-MiniLM-L6-v2",
            embedding_dimension=384,
            provider_id="faiss",
        )
        
        documents = [
            Document(
                document_id="migration-1",
                content="Migration in software systems involves moving from one approach to another while maintaining functionality and improving performance or capabilities.",
                mime_type="text/plain",
                metadata={"topic": "migration", "complexity": "basic"}
            ),
            Document(
                document_id="migration-2",
                content="When migrating RAG systems, consider factors like performance requirements, complexity needs, and integration patterns with existing systems.",
                mime_type="text/plain",
                metadata={"topic": "rag-migration", "complexity": "intermediate"}
            )
        ]
        
        self.client.tool_runtime.rag_tool.insert(
            documents=documents,
            vector_db_id=self.vector_db_id,
            chunk_size_in_tokens=256
        )
        
        cprint("✓ Demo data ready", "green")

    def migration_1_vector_io_to_rag_tool(self):
        """Migration 1: Vector IO → RAG Tool API"""
        cprint("\n" + "="*80, "yellow")
        cprint("MIGRATION 1: Vector IO → RAG Tool API", "yellow")
        cprint("="*80, "yellow")
        cprint("Reason: Add query processing and context formatting", "white")
        
        query = "How to migrate RAG systems?"
        
        # BEFORE: Vector IO approach
        cprint("\n📦 BEFORE: Vector IO API (Manual Processing)", "red")
        cprint("Code pattern: Manual chunk processing and context assembly", "gray")
        
        vector_response = self.client.vector_io.query_chunks(
            vector_db_id=self.vector_db_id,
            query=[{"type": "text", "text": query}],
            params={"max_chunks": 3, "mode": "vector"}
        )
        
        # Manual context assembly (what you'd have to do with Vector IO)
        manual_context = "Retrieved information:\n"
        for i, chunk in enumerate(vector_response.chunks):
            manual_context += f"{i+1}. {chunk.content}\n"
        
        cprint(f"Manual context assembly required:", "white")
        cprint(f"{manual_context[:200]}...", "gray")
        
        # AFTER: RAG Tool approach
        cprint("\n🔧 AFTER: RAG Tool API (Automatic Processing)", "green")
        cprint("Code pattern: Automatic query processing and formatting", "gray")
        
        rag_result = self.client.tool_runtime.rag_tool.query(
            content=[{"type": "text", "text": query}],
            vector_db_ids=[self.vector_db_id],
            query_config={
                "max_chunks": 3,
                "chunk_template": "Source {index}: {chunk.content}\n",
                "query_generator_config": {"type": "default"}
            }
        )
        
        if rag_result.content:
            formatted_content = ""
            for item in rag_result.content:
                if hasattr(item, 'text'):
                    formatted_content += item.text
            cprint(f"Automatically formatted context:", "white")
            cprint(f"{formatted_content[:200]}...", "gray")
        
        # Migration code example
        cprint("\n📝 Migration Code Pattern:", "blue")
        print("""
# BEFORE: Vector IO (manual processing)
response = client.vector_io.query_chunks(vector_db_id, query, params)
context = ""
for i, chunk in enumerate(response.chunks):
    context += f"Result {i+1}: {chunk.content}\\n"

# AFTER: RAG Tool (automatic processing)  
result = client.tool_runtime.rag_tool.query(
    content=query,
    vector_db_ids=[vector_db_id],
    query_config=RAGQueryConfig(
        max_chunks=params["max_chunks"],
        chunk_template="Result {index}: {chunk.content}\\n"
    )
)
context = "".join(item.text for item in result.content if hasattr(item, 'text'))
        """)

    def migration_2_rag_tool_to_agent(self):
        """Migration 2: RAG Tool API → Agent-based RAG"""
        cprint("\n" + "="*80, "yellow")
        cprint("MIGRATION 2: RAG Tool API → Agent-based RAG", "yellow")
        cprint("="*80, "yellow")
        cprint("Reason: Add conversational context and automatic invocation", "white")
        
        query = "What should I consider when migrating?"
        
        # BEFORE: RAG Tool approach
        cprint("\n🔧 BEFORE: RAG Tool API (Manual Invocation)", "red")
        cprint("Code pattern: Manual RAG queries + manual LLM calls", "gray")
        
        rag_result = self.client.tool_runtime.rag_tool.query(
            content=[{"type": "text", "text": query}],
            vector_db_ids=[self.vector_db_id],
            query_config={"max_chunks": 2}
        )
        
        # Manual LLM call with context (what you'd do with RAG Tool)
        context = ""
        if rag_result.content:
            for item in rag_result.content:
                if hasattr(item, 'text'):
                    context += item.text
        
        cprint("Manual steps required:", "white")
        cprint("1. Call RAG Tool for context", "gray")
        cprint("2. Manually format prompt with context", "gray")
        cprint("3. Call LLM inference separately", "gray")
        cprint("4. Handle conversation state manually", "gray")
        
        # AFTER: Agent approach
        cprint("\n🤖 AFTER: Agent-based RAG (Automatic Integration)", "green")
        cprint("Code pattern: Agent automatically handles RAG + LLM + conversation", "gray")
        
        agent_config = {
            "model": self.model_id,
            "instructions": "You are a helpful assistant. Use retrieved context to answer questions.",
            "toolgroups": [{
                "name": "builtin::rag",
                "args": {
                    "vector_db_ids": [self.vector_db_id],
                    "query_config": {"max_chunks": 2}
                }
            }]
        }
        
        agent = Agent(self.client, agent_config)
        session_id = agent.create_session("migration_demo")
        
        cprint("Automatic steps handled by agent:", "white")
        cprint("1. ✓ Determines when RAG is needed", "gray")
        cprint("2. ✓ Calls RAG Tool automatically", "gray")
        cprint("3. ✓ Integrates context with response", "gray")
        cprint("4. ✓ Manages conversation state", "gray")
        
        # Migration code example
        cprint("\n📝 Migration Code Pattern:", "blue")
        print("""
# BEFORE: RAG Tool (manual integration)
rag_result = client.tool_runtime.rag_tool.query(content, vector_db_ids, config)
context = extract_context(rag_result)
prompt = f"Context: {context}\\nQuestion: {user_query}"
response = client.inference.chat_completion(model, [{"role": "user", "content": prompt}])

# AFTER: Agent (automatic integration)
agent_config = {
    "model": model,
    "toolgroups": [{"name": "builtin::rag", "args": {"vector_db_ids": vector_db_ids}}]
}
agent = Agent(client, agent_config)
response = agent.create_turn([{"role": "user", "content": user_query}], session_id)
        """)

    def migration_3_agent_to_rag_tool(self):
        """Migration 3: Agent-based RAG → RAG Tool API"""
        cprint("\n" + "="*80, "yellow")
        cprint("MIGRATION 3: Agent-based RAG → RAG Tool API", "yellow")
        cprint("="*80, "yellow")
        cprint("Reason: Reduce overhead for performance-critical applications", "white")
        
        # BEFORE: Agent approach (higher overhead)
        cprint("\n🤖 BEFORE: Agent-based RAG (High Overhead)", "red")
        cprint("Overhead sources:", "gray")
        cprint("• Agent processing and decision making", "gray")
        cprint("• Session management", "gray")
        cprint("• Event logging and streaming", "gray")
        cprint("• Tool invocation routing", "gray")
        
        # AFTER: RAG Tool approach (lower overhead)
        cprint("\n🔧 AFTER: RAG Tool API (Reduced Overhead)", "green")
        cprint("Performance improvements:", "gray")
        cprint("• Direct RAG Tool calls", "gray")
        cprint("• No agent processing overhead", "gray")
        cprint("• Simplified execution path", "gray")
        cprint("• Manual control over LLM integration", "gray")
        
        # Migration code example
        cprint("\n📝 Migration Code Pattern:", "blue")
        print("""
# BEFORE: Agent (automatic but higher overhead)
agent_config = {"toolgroups": [{"name": "builtin::rag", "args": {...}}]}
agent = Agent(client, agent_config)
response = agent.create_turn(messages, session_id)

# AFTER: RAG Tool (manual but lower overhead)
rag_result = client.tool_runtime.rag_tool.query(content, vector_db_ids, config)
context = extract_context(rag_result)
# Manual LLM call with optimized prompt
response = client.inference.chat_completion(model, optimized_messages)
        """)

    def migration_4_rag_tool_to_vector_io(self):
        """Migration 4: RAG Tool API → Vector IO"""
        cprint("\n" + "="*80, "yellow")
        cprint("MIGRATION 4: RAG Tool API → Vector IO", "yellow")
        cprint("="*80, "yellow")
        cprint("Reason: Maximum performance and custom processing", "white")
        
        # BEFORE: RAG Tool approach
        cprint("\n🔧 BEFORE: RAG Tool API (Built-in Processing)", "red")
        cprint("Processing overhead:", "gray")
        cprint("• Query generation", "gray")
        cprint("• Context formatting", "gray")
        cprint("• Template processing", "gray")
        cprint("• Multi-database aggregation", "gray")
        
        # AFTER: Vector IO approach
        cprint("\n📦 AFTER: Vector IO API (Custom Processing)", "green")
        cprint("Performance benefits:", "gray")
        cprint("• Direct database access", "gray")
        cprint("• Custom ranking algorithms", "gray")
        cprint("• Optimized chunk processing", "gray")
        cprint("• Provider-specific optimizations", "gray")
        
        # Migration code example
        cprint("\n📝 Migration Code Pattern:", "blue")
        print("""
# BEFORE: RAG Tool (built-in processing)
result = client.tool_runtime.rag_tool.query(
    content=content,
    vector_db_ids=vector_db_ids,
    query_config=RAGQueryConfig(...)
)

# AFTER: Vector IO (custom processing)
responses = []
for vector_db_id in vector_db_ids:
    response = client.vector_io.query_chunks(
        vector_db_id=vector_db_id,
        query=content,
        params=custom_params
    )
    responses.append(response)

# Custom aggregation and ranking
chunks = custom_aggregate_and_rank(responses)
context = custom_format_context(chunks)
        """)

    def decision_matrix(self):
        """Help users decide which method to use"""
        cprint("\n" + "="*80, "cyan")
        cprint("DECISION MATRIX: CHOOSING THE RIGHT METHOD", "cyan")
        cprint("="*80, "cyan")
        
        scenarios = [
            {
                "scenario": "Building a chatbot",
                "requirements": ["Conversational context", "Session persistence", "Automatic RAG"],
                "recommended": "Agent-based RAG",
                "reason": "Handles conversation flow automatically"
            },
            {
                "scenario": "API endpoint for RAG queries",
                "requirements": ["Standardized processing", "Good performance", "Simple integration"],
                "recommended": "RAG Tool API", 
                "reason": "Balanced features and performance"
            },
            {
                "scenario": "High-performance search service",
                "requirements": ["Maximum speed", "Custom ranking", "Minimal overhead"],
                "recommended": "Vector IO API",
                "reason": "Direct database access, no processing overhead"
            },
            {
                "scenario": "Research prototype",
                "requirements": ["Custom algorithms", "Experimental features", "Full control"],
                "recommended": "Vector IO API",
                "reason": "Complete control over all processing steps"
            },
            {
                "scenario": "Enterprise application",
                "requirements": ["Reliability", "Standard features", "Good documentation"],
                "recommended": "RAG Tool API",
                "reason": "Production-ready with standard RAG features"
            }
        ]
        
        for scenario in scenarios:
            cprint(f"\n📋 Scenario: {scenario['scenario']}", "yellow")
            cprint(f"Requirements: {', '.join(scenario['requirements'])}", "white")
            cprint(f"Recommended: {scenario['recommended']}", "green")
            cprint(f"Reason: {scenario['reason']}", "gray")

    def migration_checklist(self):
        """Provide a checklist for migrations"""
        cprint("\n" + "="*80, "cyan")
        cprint("MIGRATION CHECKLIST", "cyan")
        cprint("="*80, "cyan")
        
        cprint("\n✅ Pre-Migration Steps:", "yellow")
        checklist_pre = [
            "Identify current performance bottlenecks",
            "Document existing functionality requirements",
            "Measure current latency and throughput",
            "Review integration points with other systems",
            "Plan for backward compatibility if needed"
        ]
        
        for item in checklist_pre:
            cprint(f"□ {item}", "white")
        
        cprint("\n✅ During Migration:", "yellow")
        checklist_during = [
            "Implement new method alongside existing one",
            "Create feature parity tests",
            "Benchmark performance differences",
            "Test error handling and edge cases",
            "Validate output quality and consistency"
        ]
        
        for item in checklist_during:
            cprint(f"□ {item}", "white")
        
        cprint("\n✅ Post-Migration:", "yellow")
        checklist_post = [
            "Monitor performance metrics",
            "Collect user feedback",
            "Document new integration patterns",
            "Update team training materials",
            "Plan for future optimizations"
        ]
        
        for item in checklist_post:
            cprint(f"□ {item}", "white")


def main():
    """Main migration guide demonstration"""
    guide = RAGMigrationGuide()
    
    try:
        # Setup
        guide.setup_demo_data()
        
        # Show all migration paths
        guide.migration_1_vector_io_to_rag_tool()
        guide.migration_2_rag_tool_to_agent()
        guide.migration_3_agent_to_rag_tool()
        guide.migration_4_rag_tool_to_vector_io()
        
        # Decision support
        guide.decision_matrix()
        guide.migration_checklist()
        
        cprint("\n✅ Migration guide completed!", "green")
        cprint("\nKey Migration Principles:", "yellow")
        cprint("1. Start with your performance and feature requirements", "white")
        cprint("2. Each method builds on the previous one", "white")
        cprint("3. You can migrate in either direction as needs change", "white")
        cprint("4. Test thoroughly when changing methods", "white")
        cprint("5. Consider the total cost of ownership", "white")
        
    except Exception as e:
        cprint(f"\n❌ Migration guide failed: {str(e)}", "red")
        cprint("Make sure Llama Stack server is running", "yellow")


if __name__ == "__main__":
    main()