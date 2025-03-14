# Agent Orchestration System

This project implements a sophisticated agent orchestration system using llama-stack, as described in the architecture sequence diagram. The system includes:

- **Orchestrator Agent**: Coordinates task planning and delegation
- **Self-Play Customization**: Creates and customizes agents dynamically
- **Specialized Agents**: Web, Code, File, and RAG agents with specific capabilities
- **Tool Systems**: Integration with llama-stack's tool system
- **Memory Systems**: Short-term memory and vector database for long-term storage
- **Environment Interaction**: Various interfaces to external systems

## Architecture

The system follows this high-level flow:

1. User submits a request in natural language
2. Orchestrator analyzes the request and breaks it into subtasks
3. If needed, Self-Play customization creates new agents or tools
4. Tasks are assigned to specialized agents
5. Agents execute tasks using appropriate tools
6. Results are compiled and returned to the user

## Components

- `orchestrator.py`: Main orchestration logic with task planning, delegation, and execution
- `specialized_agents.py`: Definitions for various specialized agent types
- `memory.py` (planned): Enhanced memory system for long-term storage and retrieval
- `tools.py` (planned): Custom tool definitions for specific tasks

## Prerequisites

- Python 3.9+
- An operational llama-stack instance
- Environment variables:
  - `LLAMA_STACK_PORT`: Port for the llama-stack instance (default: 8080)
  - `INFERENCE_MODEL`: LLM model ID to use (e.g., "llama3")

## Setup

1. Ensure llama-stack is running and accessible
2. Set the required environment variables
3. Install required dependencies:

```bash
pip install llama-stack-client termcolor
```

## Running the System

To run the system:

```bash
cd experiments/agent-orchestration-system
python orchestrator.py
```

## Example Usage

The orchestrator's `process_request` method takes a natural language request and processes it through the system:

```python
response = await orchestrator.process_request("Research AAPL and MSFT stocks and create a comparison chart")
```

## Customization

You can define custom agent types by extending the `BaseAgent` class in `specialized_agents.py` or by using the `SelfPlayCustomization` system to dynamically create agents based on requirements.

## Planned Enhancements

- Enhanced memory system with vector database integration
- Custom tool development for specialized tasks
- User interface for interacting with the system
- Enhanced error handling and recovery mechanisms
- Session persistence for long-running agent conversations 