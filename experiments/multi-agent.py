from llama_stack_client import LlamaStackClient
from llama_stack.distribution.library_client import LlamaStackAsLibraryClient
from llama_stack_client.lib.agents.agent import Agent
from rich.pretty import pprint
import json
import uuid
from pydantic import BaseModel
import rich
import os

# Set up environment variables for API keys
# You would need to set your API key here
# os.environ['FIREWORKS_API_KEY'] = "your-api-key-here"

# Initialize client - choose one of these options:
# Option 1: Use LlamaStackAsLibraryClient (for local usage)
client = LlamaStackAsLibraryClient("fireworks", provider_data={"fireworks_api_key": os.environ.get('FIREWORKS_API_KEY')})
_ = client.initialize()

# Option 2: Use LlamaStackClient (for connecting to a hosted server)
# client = LlamaStackClient(base_url="http://localhost:8321")

# Define the model to use
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"

# Base configuration for all agents
base_agent_config = dict(
    model=MODEL_ID,
    instructions="You are a helpful assistant.",
    sampling_params={
        "strategy": {"type": "top_p", "temperature": 1.0, "top_p": 0.9},
    },
)

# 1. Define specialized agents
billing_agent_config = {
    **base_agent_config,
    "instructions": """You are a billing support specialist. Follow these guidelines:
    1. Always start with "Billing Support Response:"
    2. First acknowledge the specific billing issue
    3. Explain any charges or discrepancies clearly
    4. List concrete next steps with timeline
    5. End with payment options if relevant
    
    Keep responses professional but friendly.
    """,
}

technical_agent_config = {
    **base_agent_config,
    "instructions": """You are a technical support engineer. Follow these guidelines:
    1. Always start with "Technical Support Response:"
    2. List exact steps to resolve the issue
    3. Include system requirements if relevant
    4. Provide workarounds for common problems
    5. End with escalation path if needed
    
    Use clear, numbered steps and technical details.
    """,
}

account_agent_config = {
    **base_agent_config,
    "instructions": """You are an account security specialist. Follow these guidelines:
    1. Always start with "Account Support Response:"
    2. Prioritize account security and verification
    3. Provide clear steps for account recovery/changes
    4. Include security tips and warnings
    5. Set clear expectations for resolution time
    
    Maintain a serious, security-focused tone.
    """,
}

product_agent_config = {
    **base_agent_config,
    "instructions": """You are a product specialist. Follow these guidelines:
    1. Always start with "Product Support Response:"
    2. Focus on feature education and best practices
    3. Include specific examples of usage
    4. Link to relevant documentation sections
    5. Suggest related features that might help
    
    Be educational and encouraging in tone.
    """,
}

# Create specialized agent instances
specialized_agents = {
    "billing": Agent(client, **billing_agent_config),
    "technical": Agent(client, **technical_agent_config),
    "account": Agent(client, **account_agent_config),
    "product": Agent(client, **product_agent_config),
}

# 2. Define a routing agent with output schema
class OutputSchema(BaseModel):
    reasoning: str
    support_team: str

routing_agent_config = {
    **base_agent_config,
    "instructions": f"""You are a routing agent. Analyze the user's input and select the most appropriate support team from these options: 

    {list(specialized_agents.keys())}

    Return the name of the support team in JSON format.

    First explain your reasoning, then provide your selection in this JSON format: 
    {{
        "reasoning": "<your explanation>",
        "support_team": "<support team name>"
    }}

    Note the support team name can only be one of the following: {specialized_agents.keys()}
    """,
    "response_format": {
        "type": "json_schema",
        "json_schema": OutputSchema.model_json_schema()
    }
}

routing_agent = Agent(client, **routing_agent_config)

# 3. Create a session for all agents
routing_agent_session_id = routing_agent.create_session(session_name=f"routing_agent_{uuid.uuid4()}")
specialized_agents_session_ids = {
    "billing": specialized_agents["billing"].create_session(session_name=f"billing_agent_{uuid.uuid4()}"),
    "technical": specialized_agents["technical"].create_session(session_name=f"technical_agent_{uuid.uuid4()}"),
    "account": specialized_agents["account"].create_session(session_name=f"account_agent_{uuid.uuid4()}"),
    "product": specialized_agents["product"].create_session(session_name=f"product_agent_{uuid.uuid4()}"),
}

# 4. Combine routing with specialized agents
def process_user_query(query):
    # Step 1: Route to the appropriate support team
    routing_response = routing_agent.create_turn(
        messages=[
            {
                "role": "user",
                "content": query,
            }
        ],
        session_id=routing_agent_session_id,
        stream=False,
    )
    try:
        routing_result = json.loads(routing_response.output_message.content)
        rich.print(f"🔀 [cyan] Routing Result: {routing_result['reasoning']} [/cyan]")
        rich.print(f"🔀 [cyan] Routing to {routing_result['support_team']}... [/cyan]")

        # Route to the appropriate support team
        return specialized_agents[routing_result["support_team"]].create_turn(
            messages=[
                {"role": "user", "content": query}
            ],
            session_id=specialized_agents_session_ids[routing_result["support_team"]],
            stream=False,
        )
    except json.JSONDecodeError:
        print("Error: Invalid JSON response from routing agent")
        return None

# Example support tickets to process
tickets = [
    """Subject: Can't access my account
    Message: Hi, I've been trying to log in for the past hour but keep getting an 'invalid password' error. 
    I'm sure I'm using the right password. Can you help me regain access? This is urgent as I need to 
    submit a report by end of day.
    - John""",
    
    """Subject: Unexpected charge on my card
    Message: Hello, I just noticed a charge of $49.99 on my credit card from your company, but I thought
    I was on the $29.99 plan. Can you explain this charge and adjust it if it's a mistake?
    Thanks,
    Sarah""",
    
    """Subject: How to export data?
    Message: I need to export all my project data to Excel. I've looked through the docs but can't
    figure out how to do a bulk export. Is this possible? If so, could you walk me through the steps?
    Best regards,
    Mike"""
]

# Process each ticket through the multi-agent system
def main():
    for i, ticket in enumerate(tickets):
        print(f"========= Processing ticket {i+1}: =========")
        response = process_user_query(ticket)
        print(response.output_message.content)
        print("\n")

    # Optional: Examine the internal details of agent sessions
    print("Routing Agent Session:")
    routing_agent_session = client.agents.session.retrieve(session_id=routing_agent_session_id, agent_id=routing_agent.agent_id)
    pprint(routing_agent_session.to_dict())

    for specialized_agent_type, specialized_agent in specialized_agents.items():
        specialized_agent_session = client.agents.session.retrieve(session_id=specialized_agent.session_id, agent_id=specialized_agent.agent_id)
        print(f"Specialized Agent {specialized_agent_type} Session:")
        pprint(specialized_agent_session.to_dict())

if __name__ == "__main__":
    main()
