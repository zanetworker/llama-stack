"""
Specialized Agents for the Agent Orchestration System

This module defines specialized agent types that can be used in the system:
- WebAgent: For web search and API interactions
- CodeAgent: For code generation, analysis and execution
- FileAgent: For file operations
- CustomAgent: Base for dynamically created agents
"""

from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass

from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.client_tool import client_tool
from llama_stack_client.types.agent_create_params import AgentConfig

logger = logging.getLogger(__name__)

@dataclass
class ToolDefinition:
    """Definition of a tool for an agent"""
    name: str
    description: str
    args: Dict[str, Any] = None
    
    def to_toolgroup(self) -> Dict[str, Any]:
        """Convert to toolgroup format"""
        return {
            "name": self.name,
            "args": self.args or {}
        }

class BaseAgent:
    """Base class for all agent types"""
    
    def __init__(self, client: LlamaStackClient, model_id: str, agent_id: str):
        self.client = client
        self.model_id = model_id
        self.agent_id = agent_id
        self.agent = None
        self.session_id = None
        self.tools = []
        
    def get_instructions(self) -> str:
        """Get instructions for this agent type"""
        return """You are a helpful assistant."""
    
    def get_tools(self) -> List[ToolDefinition]:
        """Get tools for this agent type"""
        return []
    
    def create_instance(self) -> tuple[Agent, str]:
        """Create an instance of this agent type"""
        toolgroups = [tool.to_toolgroup() for tool in self.get_tools()]
        
        config = AgentConfig(
            model=self.model_id,
            instructions=self.get_instructions(),
            toolgroups=toolgroups,
            enable_session_persistence=True,
            tool_choice="auto"
        )
        
        self.agent = Agent(self.client, config)
        self.session_id = self.agent.create_session(f"{self.agent_id}-session")
        
        logger.info(f"Created {self.__class__.__name__} instance with session {self.session_id}")
        return self.agent, self.session_id
    
    def get_agent_instance(self) -> tuple[Agent, str]:
        """Get the agent instance, creating it if needed"""
        if not self.agent or not self.session_id:
            return self.create_instance()
        return self.agent, self.session_id


class WebAgent(BaseAgent):
    """Agent specialized in web search and retrieval"""
    
    def get_instructions(self) -> str:
        return """You are a specialized web agent capable of searching the internet and retrieving information.
        
Your capabilities:
1. Search the web for up-to-date information on any topic
2. Extract relevant data from web pages
3. Answer questions based on web information
4. Summarize content from websites
5. Retrieve specific facts and statistics

Guidelines:
- Always cite your sources by including URLs
- Be concise and focus on the most relevant information
- When uncertain about information, acknowledge the limitations
- Present information in a structured, easy-to-understand format
- Prioritize reliable and authoritative sources
"""
    
    def get_tools(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="builtin::websearch",
                description="Search the web for information",
                args={}
            )
        ]


class CodeAgent(BaseAgent):
    """Agent specialized in code generation and analysis"""
    
    def get_instructions(self) -> str:
        return """You are a specialized code agent with expertise in programming, software development, and technical problem-solving.
        
Your capabilities:
1. Generate code snippets and complete solutions in various programming languages
2. Debug and fix issues in existing code
3. Explain programming concepts and techniques
4. Provide code reviews and optimization suggestions
5. Help with algorithm design and implementation

Guidelines:
- Provide well-commented, readable code that follows best practices
- Explain your code when it would be helpful for understanding
- Consider edge cases and potential errors in your solutions
- Structure your responses with clear explanations before and after code blocks
- When suggesting improvements, explain the reasoning behind them
"""
    
    def get_tools(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="builtin::code_interpreter",
                description="Execute and interpret code",
                args={}
            )
        ]


class FileAgent(BaseAgent):
    """Agent specialized in file operations"""
    
    def get_instructions(self) -> str:
        return """You are a specialized file agent with capabilities for managing, processing, and analyzing files.
        
Your capabilities:
1. Read from and write to files
2. Process and transform file contents
3. Extract information from files
4. Handle various file formats (text, CSV, JSON, etc.)
5. Organize and manage file structures

Guidelines:
- Always confirm file operations before making changes
- Be careful with destructive operations (deleting, overwriting)
- Handle errors gracefully when files don't exist or are inaccessible
- Process files efficiently, especially for large files
- Provide clear summaries of file operations performed
"""
    
    def get_tools(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="builtin::filesystem",
                description="Interact with the file system",
                args={}
            )
        ]


class RAGAgent(BaseAgent):
    """Agent specialized in retrieval-augmented generation"""
    
    def __init__(self, client: LlamaStackClient, model_id: str, agent_id: str, vector_db_id: str):
        super().__init__(client, model_id, agent_id)
        self.vector_db_id = vector_db_id
    
    def get_instructions(self) -> str:
        return f"""You are a specialized RAG (Retrieval-Augmented Generation) agent with access to a knowledge base.
        
Your capabilities:
1. Search the knowledge base for relevant information
2. Provide answers based on retrieved information
3. Cite sources from the knowledge base
4. Handle queries requiring specific domain knowledge
5. Acknowledge when information is not available in the knowledge base

Guidelines:
- Base your answers on the retrieved information from vector database {self.vector_db_id}
- Do not make up information if it's not in the retrieved results
- Indicate when you're not certain or when information might be incomplete
- Structure responses clearly, distinguishing between different sources
- Always provide context-appropriate responses
"""
    
    def get_tools(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="builtin::rag",
                description="Retrieve information from the knowledge base",
                args={
                    "vector_db_ids": [self.vector_db_id],
                    "query_config": {
                        "max_chunks": 5,
                        "similarity_threshold": 0.7
                    }
                }
            )
        ]


class MultiToolAgent(BaseAgent):
    """Agent with multiple tool capabilities"""
    
    def __init__(self, client: LlamaStackClient, model_id: str, agent_id: str, 
                 instructions: str, tools: List[ToolDefinition]):
        super().__init__(client, model_id, agent_id)
        self._instructions = instructions
        self._tools = tools
    
    def get_instructions(self) -> str:
        return self._instructions
    
    def get_tools(self) -> List[ToolDefinition]:
        return self._tools


def get_agent_factory(agent_type: str):
    """Get agent factory function by type"""
    agent_types = {
        "WebAgent": WebAgent,
        "CodeAgent": CodeAgent,
        "FileAgent": FileAgent,
        "RAGAgent": RAGAgent,
    }
    
    return agent_types.get(agent_type, BaseAgent)


def create_agent_by_profile(client: LlamaStackClient, model_id: str, profile: Dict[str, Any]) -> BaseAgent:
    """Create an agent based on a profile"""
    agent_type = profile.get("agent_type", "BaseAgent")
    agent_id = profile.get("agent_id", "custom_agent")
    instructions = profile.get("instructions", "")
    required_tools = profile.get("required_tools", [])
    
    # Map tool names to tool definitions
    tools = []
    for tool_name in required_tools:
        if tool_name == "web_search":
            tools.append(ToolDefinition(
                name="builtin::websearch",
                description="Search the web for information",
                args={}
            ))
        elif tool_name == "code_interpreter":
            tools.append(ToolDefinition(
                name="builtin::code_interpreter",
                description="Execute and interpret code",
                args={}
            ))
        elif tool_name == "filesystem":
            tools.append(ToolDefinition(
                name="builtin::file_system",
                description="Interact with the file system",
                args={}
            ))
        elif tool_name == "rag":
            tools.append(ToolDefinition(
                name="builtin::rag",
                description="Retrieve information from knowledge base",
                args={"vector_db_ids": ["orchestrator-memory"]}
            ))
    
    # Create appropriate agent type
    if agent_type in ["WebAgent", "CodeAgent", "FileAgent"]:
        agent_factory = get_agent_factory(agent_type)
        return agent_factory(client, model_id, agent_id)
    else:
        # Default to multi-tool agent for custom types
        return MultiToolAgent(client, model_id, agent_id, instructions, tools) 