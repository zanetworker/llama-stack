import asyncio
import fire
import os
import logging

from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.client_tool import client_tool
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from termcolor import colored


# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@client_tool
def calculate(x: float, y: float, operation: str) -> dict:
    """Simple calculator tool that performs basic math operations.

    :param x: First number to perform operation on
    :param y: Second number to perform operation on
    :param operation: Mathematical operation to perform ('add', 'subtract', 'multiply', 'divide')
    :returns: Dictionary containing success status and result or error message
    """
    logger.debug(f"Calculator called with: x={x}, y={y}, operation={operation}")
    try:
        if operation == "add":
            result = x + y
        elif operation == "subtract":
            result = x - y
        elif operation == "multiply":
            result = x * y
        elif operation == "divide":
            if y == 0:
                return {"success": False, "error": "Cannot divide by zero"}
            result = x / y
        else:
            return {"success": False, "error": "Invalid operation"}

        logger.debug(f"Calculator result: {result}")
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"Calculator error: {str(e)}")
        return {"success": False, "error": str(e)}


async def run_main():
    client = LlamaStackClient(
        base_url=f"http://localhost:{os.environ.get('LLAMA_STACK_PORT', '8080')}"
    )

    logger.debug("Setting up agent config...")

    print("Calculate tool definition:")
    print(calculate.get_tool_definition())
    available_models = [
        model.identifier for model in client.models.list() if model.model_type == "llm"
    ]
    if not available_models:
        print(colored("No available models. Exiting.", "red"))
        return
    else:
        selected_model = available_models[0]
        print(f"Using model: {selected_model}")

    agent_config = AgentConfig(
        # model=os.environ.get('INFERENCE_MODEL', 'llama3.2:3b-instruct-fp16'),
        model=selected_model,
        instructions="""You are a calculator assistant. Use the calculate tool to perform operations.
When using the calculate tool:
1. Extract the numbers and operation from the user's request
2. Use the appropriate operation (add, subtract, multiply, divide)
3. Present the result clearly
4. Handle any errors gracefully""",
        client_tools=[calculate.get_tool_definition()],
        toolgroups=[],
        tool_choice="auto",
        enable_session_persistence=False,
        tool_prompt_format="python_list",
        max_tool_calls=3,
    )
    logger.debug(f"Agent config: {agent_config}")

    agent = Agent(client, agent_config, (calculate,))
    session_id = agent.create_session("calc-session")
    logger.debug(f"Created session: {session_id}")

    prompt = "What is 25 plus 15?"
    print(f"\nUser: {prompt}")

    try:
        response_gen = agent.create_turn(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            session_id=session_id,
        )
        print("Agent response:")
        print(response_gen)

        event_logger = EventLogger()
        for chunk in event_logger.log(response_gen):
            if chunk is None:
                continue
            try:
                chunk.print(flush=True)
            except Exception as e:
                logger.error(f"Error printing chunk: {e}")
                continue

    except Exception as e:
        logger.error(f"Error during agent turn: {e}")
        raise


def main():
    asyncio.run(run_main())


if __name__ == "__main__":
    fire.Fire(main)