"""
Agent Orchestration System using llama-stack

This file implements a sophisticated agent orchestration system with:
- Orchestrator Agent
- Self-Play Customization
- Specialized Agents 
- Tool Systems
- Memory Systems
- Environment interaction

Based on the architecture in the sequence diagram.
"""

import os
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union
from termcolor import cprint
from dataclasses import dataclass, field

from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.lib.agents.client_tool import client_tool
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types import Document, UserMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type definitions for our system
@dataclass
class Task:
    """A task to be executed by an agent"""
    task_id: str
    description: str
    status: str = "pending"  # pending, in_progress, completed, failed
    subtasks: List['Task'] = field(default_factory=list)
    result: Optional[Any] = None
    agent_id: Optional[str] = None

@dataclass
class Memory:
    """Memory store for the system"""
    short_term: Dict[str, Any] = field(default_factory=dict)
    vector_db_id: Optional[str] = None
    
    def store(self, key: str, value: Any) -> None:
        """Store a value in short-term memory"""
        self.short_term[key] = value
        
    def retrieve(self, key: str) -> Any:
        """Retrieve a value from short-term memory"""
        return self.short_term.get(key)
    
    def list_keys(self) -> List[str]:
        """List all keys in short-term memory"""
        return list(self.short_term.keys())

class OrchestratorAgent:
    """The main orchestrator that delegates tasks to specialized agents"""
    
    def __init__(self, client: LlamaStackClient, model_id: str):
        self.client = client
        self.model_id = model_id
        self.memory = Memory(vector_db_id="orchestrator-memory")
        self.specialized_agents = {}
        self.agent_instances = {}
        self.setup_vector_store()
        
    def setup_vector_store(self):
        """Initialize vector database for memory"""
        try:
            # Clean up existing vector db if it exists
            try:
                self.client.vector_dbs.delete(vector_db_id=self.memory.vector_db_id)
            except:
                pass

            # Register new vector db
            self.client.vector_dbs.register(
                vector_db_id=self.memory.vector_db_id,
                embedding_model="all-MiniLM-L6-v2",
                embedding_dimension=384,
                provider_id="faiss"
            )
            logger.info(f"Vector store setup complete: {self.memory.vector_db_id}")
        except Exception as e:
            logger.error(f"Error setting up vector store: {e}")
            raise
    
    def register_specialized_agent(self, agent_id: str, agent_type: str, instructions: str, 
                                  toolgroups: List[Dict[str, Any]] = None, 
                                  client_tools: List[Dict[str, Any]] = None):
        """Register a specialized agent with the orchestrator"""
        if not toolgroups:
            toolgroups = []
            
        if not client_tools:
            client_tools = []
            
        self.specialized_agents[agent_id] = {
            "agent_type": agent_type,
            "instructions": instructions,
            "toolgroups": toolgroups,
            "client_tools": client_tools
        }
        logger.info(f"Registered {agent_type} agent: {agent_id}")
        
    def create_agent_instance(self, agent_id: str) -> str:
        """Create an instance of a specialized agent"""
        if agent_id not in self.specialized_agents:
            raise ValueError(f"Agent {agent_id} not registered")
        
        agent_config = self.specialized_agents[agent_id]
        
        config = AgentConfig(
            model=self.model_id,
            instructions=agent_config["instructions"],
            toolgroups=agent_config["toolgroups"],
            client_tools=agent_config["client_tools"],
            enable_session_persistence=True,
            tool_choice="auto"
        )
        
        agent = Agent(self.client, config)
        session_id = agent.create_session(f"{agent_id}-session")
        
        self.agent_instances[agent_id] = {
            "agent": agent,
            "session_id": session_id
        }
        
        logger.info(f"Created agent instance for {agent_id} with session {session_id}")
        return session_id
    
    def get_agent_instance(self, agent_id: str) -> tuple[Agent, str]:
        """Get an agent instance by ID, creating it if it doesn't exist"""
        if agent_id not in self.agent_instances:
            self.create_agent_instance(agent_id)
            
        instance = self.agent_instances[agent_id]
        return instance["agent"], instance["session_id"]
    
    def parse_task(self, description: str) -> List[Task]:
        """Use the LLM to parse a high-level task into subtasks"""
        prompt = f"""
        Parse the following task into a well-structured sequence of subtasks.
        For each subtask, identify which type of agent would be best suited to handle it:
        - WebAgent: For tasks requiring web search or API calls
        - CodeAgent: For tasks involving coding, debugging, or analysis
        - FileAgent: For tasks involving file reading, writing, or manipulation
        
        Task: {description}
        
        Return your answer as a JSON array of subtasks with the following format:
        [
            {{
                "task_id": "unique_id",
                "description": "Detailed description of the subtask",
                "agent_type": "One of: WebAgent, CodeAgent, FileAgent"
            }},
            ...
        ]
        """
        
        response = self.client.inference.chat_completion(
            model_id=self.model_id,
            messages=[
                {"role": "system", "content": "You are a task planning assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        
        content = response.completion_message.content
        
        # Extract JSON from response
        try:
            # Find JSON array in the response
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                subtasks_data = json.loads(json_str)
            else:
                # Fallback if JSON not properly formatted
                subtasks_data = [
                    {
                        "task_id": "task_1",
                        "description": description,
                        "agent_type": "CodeAgent"
                    }
                ]
            
            # Convert to Task objects
            subtasks = []
            for i, data in enumerate(subtasks_data):
                task_id = data.get("task_id", f"task_{i+1}")
                subtasks.append(Task(
                    task_id=task_id,
                    description=data["description"],
                    agent_id=self._map_agent_type_to_id(data["agent_type"])
                ))
            
            return subtasks
        except Exception as e:
            logger.error(f"Error parsing subtasks: {e}")
            # Return a single task as fallback
            return [Task(
                task_id="task_1",
                description=description,
                agent_id="code_agent"
            )]
    
    def _map_agent_type_to_id(self, agent_type: str) -> str:
        """Map agent type to agent ID"""
        mapping = {
            "WebAgent": "web_agent",
            "CodeAgent": "code_agent",
            "FileAgent": "file_agent"
        }
        return mapping.get(agent_type, "code_agent")
    
    async def execute_task(self, task: Task) -> Task:
        """Execute a task using the appropriate specialized agent"""
        logger.info(f"Executing task: {task.task_id} - {task.description}")
        
        if not task.agent_id:
            logger.warning(f"No agent assigned to task {task.task_id}, using code_agent")
            task.agent_id = "code_agent"
            
        task.status = "in_progress"
        
        try:
            # Get the agent instance
            agent, session_id = self.get_agent_instance(task.agent_id)
            
            # Execute the task
            response = agent.create_turn(
                messages=[UserMessage(role="user", content=task.description)],
                session_id=session_id
            )
            
            # Process the response
            full_response = ""
            for log in EventLogger().log(response):
                if log and hasattr(log, 'content') and log.content:
                    full_response += log.content
            
            # Store the result
            task.result = full_response
            task.status = "completed"
            
            # Store in memory
            self.memory.store(f"task_result_{task.task_id}", full_response)
            
            return task
        except Exception as e:
            logger.error(f"Error executing task {task.task_id}: {e}")
            task.status = "failed"
            task.result = str(e)
            return task
    
    async def execute_tasks(self, tasks: List[Task]) -> List[Task]:
        """Execute a list of tasks in sequence"""
        results = []
        for task in tasks:
            task = await self.execute_task(task)
            results.append(task)
        return results
    
    async def process_request(self, request: str) -> str:
        """Process a user request through the orchestration system"""
        logger.info(f"Processing request: {request}")
        
        # 1. Parse the request into tasks
        tasks = self.parse_task(request)
        
        # 2. Execute the tasks
        completed_tasks = await self.execute_tasks(tasks)
        
        # 3. Compile the results
        results = []
        for task in completed_tasks:
            results.append(f"Task {task.task_id}: {task.description}")
            results.append(f"Status: {task.status}")
            results.append(f"Result: {task.result}")
            results.append("---")
        
        final_response = "\n".join(results)
        
        # 4. Store the final result in memory
        self.memory.store("last_request", request)
        self.memory.store("last_response", final_response)
        
        return final_response


class SelfPlayCustomization:
    """System for creating and customizing agents dynamically"""
    
    def __init__(self, client: LlamaStackClient, model_id: str):
        self.client = client
        self.model_id = model_id
    
    def generate_agent_profile(self, requirements: str) -> Dict[str, Any]:
        """Generate a profile for a new agent based on requirements"""
        prompt = f"""
        Generate a profile for a new agent based on the following requirements:
        
        {requirements}
        
        The profile should include:
        1. A descriptive agent_id
        2. Detailed instructions for the agent
        3. Required tools and capabilities
        
        Return your response as a JSON object with the following structure:
        {{
            "agent_id": "unique_id",
            "agent_type": "One of: WebAgent, CodeAgent, FileAgent",
            "instructions": "Detailed instructions for the agent",
            "required_tools": ["tool1", "tool2", ...]
        }}
        """
        
        response = self.client.inference.chat_completion(
            model_id=self.model_id,
            messages=[
                {"role": "system", "content": "You are an agent designer assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        
        content = response.completion_message.content
        
        # Extract JSON from response
        try:
            # Find JSON in the response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                profile = json.loads(json_str)
                return profile
            else:
                # Fallback
                return {
                    "agent_id": "custom_agent",
                    "agent_type": "CodeAgent",
                    "instructions": requirements,
                    "required_tools": []
                }
        except Exception as e:
            logger.error(f"Error generating agent profile: {e}")
            return {
                "agent_id": "custom_agent",
                "agent_type": "CodeAgent",
                "instructions": requirements,
                "required_tools": []
            }
            
    def map_tools_to_toolgroups(self, required_tools: List[str]) -> List[Dict[str, Any]]:
        """Map required tools to actual toolgroups"""
        tool_mapping = {
            "rag": {
                "name": "builtin::rag",
                "args": {
                    "vector_db_ids": ["orchestrator-memory"],
                }
            },
            "web_search": {
                "name": "builtin::websearch",
                "args": {}
            },
            "code_interpreter": {
                "name": "builtin::code_interpreter",
                "args": {}
            },
            "file_system": {
                "name": "builtin::filesystem",
                "args": {}
            }
        }
        
        toolgroups = []
        for tool in required_tools:
            if tool in tool_mapping:
                toolgroups.append(tool_mapping[tool])
                
        return toolgroups
    

async def main():
    """Main entry point for the agent orchestration system"""
    # Get config from environment
    port = os.environ.get('LLAMA_STACK_PORT', '8080')
    model_id = os.environ.get('INFERENCE_MODEL', 'llama3')
    
    # Initialize client
    client = LlamaStackClient(
        base_url=f"http://localhost:{port}"
    )
    
    # Initialize orchestrator
    orchestrator = OrchestratorAgent(client, model_id)
    
    # Initialize self-play customization
    self_play = SelfPlayCustomization(client, model_id)
    
    # Register specialized agents
    orchestrator.register_specialized_agent(
        agent_id="web_agent",
        agent_type="WebAgent",
        instructions="""You are a specialized web agent. Your task is to:
        1. Search the web for information
        2. Extract relevant data from websites
        3. Present the information clearly and concisely
        """,
        toolgroups=[{
            "name": "builtin::web_search",
            "args": {}
        }]
    )
    
    orchestrator.register_specialized_agent(
        agent_id="code_agent",
        agent_type="CodeAgent",
        instructions="""You are a specialized code agent. Your task is to:
        1. Write, analyze, or debug code
        2. Explain programming concepts
        3. Provide code-related guidance
        """,
        toolgroups=[{
            "name": "builtin::code_interpreter",
            "args": {}
        }]
    )
    
    orchestrator.register_specialized_agent(
        agent_id="file_agent",
        agent_type="FileAgent",
        instructions="""You are a specialized file agent. Your task is to:
        1. Read from and write to files
        2. Process and analyze file contents
        3. Organize and manage file structures
        """,
        toolgroups=[{
            "name": "builtin::file_system",
            "args": {}
        }]
    )
    
    # Example of creating a custom agent through self-play
    custom_requirements = """
    Create an agent that can analyze financial data, specifically stock market trends.
    It should be able to fetch stock data, generate charts, and provide analysis.
    """
    
    agent_profile = self_play.generate_agent_profile(custom_requirements)
    toolgroups = self_play.map_tools_to_toolgroups(agent_profile["required_tools"])
    
    orchestrator.register_specialized_agent(
        agent_id=agent_profile["agent_id"],
        agent_type=agent_profile["agent_type"],
        instructions=agent_profile["instructions"],
        toolgroups=toolgroups
    )
    
    # Process a user request
    user_request = "Analyze AAPL and MSFT stocks over the past week and provide a comparison."
    
    cprint("User request:", "green")
    cprint(user_request, "green")
    
    response = await orchestrator.process_request(user_request)
    
    cprint("\nSystem response:", "blue")
    cprint(response, "white")


if __name__ == "__main__":
    asyncio.run(main()) 