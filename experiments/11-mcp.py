# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import os
import json
import fire
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from termcolor import colored


def get_mcp_toolgroups():
    """Define MCP tool-groups configuration for the agent."""
    return [
        {
            "name": "mcp::bewell",
            "args": {
                "url": "http://127.0.0.1:8002",
                "timeout": 30,
            }
        }
    ]


def main():
    client = LlamaStackClient(
        base_url=f"http://localhost:{os.environ.get('LLAMA_STACK_PORT', '8080')}"
    )

    available_shields = [shield.identifier for shield in client.shields.list()]
    if not available_shields:
        print(colored("No available shields. Disabling safety.", "yellow"))
    else:
        print(f"Available shields found: {available_shields}")

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
        model=selected_model,
        instructions="""You are a helpful assistant that can interact with various services through MCP tools.
When handling requests:
1. Use the appropriate MCP tool based on the user's request
2. Present the information in a clear, organized way
3. Handle any errors gracefully""",
        sampling_params={
            "strategy": {"type": "top_p", "temperature": 1.0, "top_p": 0.9},
        },
        toolgroups=get_mcp_toolgroups(),
        tool_choice="auto",
        input_shields=available_shields if available_shields else [],
        output_shields=available_shields if available_shields else [],
        enable_session_persistence=False,
    )
    agent = Agent(client, agent_config)
    
    # Example prompts for different tools
    user_prompts = [
        "get me all available meals",
        "show me my health metrics",
    ]

    session_id = agent.create_session("mcp-session")

    for prompt in user_prompts:
        print(f"\nUser: {prompt}")
        response = agent.create_turn(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            session_id=session_id,
        )

        for log in EventLogger().log(response):
            log.print()


if __name__ == "__main__":
    fire.Fire(main)
