from llama_stack_client import LlamaStackClient
from termcolor import cprint
import os
import json
from typing import List, Dict

def setup_mcp_tools(client: LlamaStackClient) -> List[Dict]:
    """Setup and return MCP tools"""
    # Check available providers
    available_providers = client.providers.list()
    cprint(f"Available providers: {[p.provider_id for p in available_providers]}", "yellow")

    # Find the MCP provider
    mcp_provider = next((p for p in available_providers if p.provider_id == "model-context-protocol"), None)
    if not mcp_provider:
        raise Exception("MCP provider not found. Please make sure it's installed and enabled.")

    cprint(f"Using MCP provider: {mcp_provider.provider_id}", "green")

    # Try to unregister the toolgroup if it exists
    try:
        client.toolgroups.unregister(toolgroup_id="mcp::docling")
    except Exception:
        pass  # Ignore if it doesn't exist

    # Register MCP tools
    try:
        client.toolgroups.register(
            toolgroup_id="mcp::docling",
            provider_id="model-context-protocol",
            mcp_endpoint={"uri": "http://0.0.0.0:8000/sse"})
        cprint("Successfully registered MCP docling toolgroup", "green")
    except Exception as e:
        raise Exception(f"Error registering MCP toolgroup: {e}")

    # Get and return available tools
    tools = client.tools.list(toolgroup_id="mcp::docling")
    if not tools:
        raise Exception("No tools found in the toolgroup")
    
    return [
        {
            "identifier": tool.identifier,
            "description": tool.description,
            "parameters": [
                {
                    "name": param.name,
                    "type": getattr(param, 'parameter_type', 'unknown'),
                    "description": getattr(param, 'description', 'No description'),
                    "required": getattr(param, 'required', False)
                }
                for param in tool.parameters
            ]
        }
        for tool in tools
    ]

def get_tool_selection(client: LlamaStackClient, task: str, available_tools: List[Dict]) -> Dict:
    """Use LLM to select appropriate tool and parameters for the task"""
    model_id = os.environ["INFERENCE_MODEL"]
    
    # Create a prompt that describes the task and available tools
    tools_description = json.dumps(available_tools, indent=2)
    prompt = f"""Given the following task and available tools, determine which tool would be most appropriate to use.
    
Task: {task}

Available Tools:
{tools_description}

You must respond with valid JSON only, in this exact format:
{{
    "selected_tool": "tool_identifier",
    "reasoning": "brief explanation of why this tool was selected",
    "parameters": {{
        "param1": "value1",
        "param2": "value2"
    }}
}}"""

    response = client.inference.chat_completion(
        model_id=model_id,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that selects appropriate tools for document processing tasks. You must respond with valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "type": "object",
                "properties": {
                    "selected_tool": {
                        "type": "string",
                        "description": "The identifier of the selected tool"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why this tool was selected"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Parameters to be passed to the tool"
                    }
                },
                "required": ["selected_tool", "reasoning", "parameters"],
                "title": "ToolSelection"
            }
        }
    )
    
    try:
        return json.loads(response.completion_message.content)
    except json.JSONDecodeError:
        cprint("Failed to parse JSON response. Using default tool selection.", "yellow")
        return {
            "selected_tool": "convert_document",
            "reasoning": "Default selection for document processing",
            "parameters": {"url": task.split(": ")[-1]}
        }

def process_result(client: LlamaStackClient, result: str, task: str) -> str:
    """Use LLM to process and summarize the tool's output"""
    model_id = os.environ["INFERENCE_MODEL"]
    
    prompt = f"""Task: {task}

Tool Output:
{str(result)[:5000]}  # Limiting to 5000 chars

Please analyze this output and provide a concise summary that addresses the original task."""

    response = client.inference.chat_completion(
        model_id=model_id,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that analyzes document processing results."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.completion_message.content

def main():
    # Initialize the client
    client = LlamaStackClient(
        base_url=f"http://localhost:{os.environ.get('LLAMA_STACK_PORT', '8080')}"
    )

    try:
        # Setup MCP tools
        available_tools = setup_mcp_tools(client)
        
        # Example tasks to process
        tasks = [
            "Convert this PDF document to markdown format: https://example.com/sample.pdf",
            "Extract all tables from this document: https://example.com/data.pdf",
            "I need to analyze the structure of this document: https://example.com/report.pdf"
        ]

        for task in tasks:
            cprint(f"\nProcessing task: {task}", "yellow")
            
            # Use LLM to select appropriate tool
            tool_selection = get_tool_selection(client, task, available_tools)
            cprint("\nTool Selection:", "cyan")
            cprint(f"Selected Tool: {tool_selection['selected_tool']}", "cyan")
            cprint(f"Reasoning: {tool_selection['reasoning']}", "cyan")
            cprint(f"Parameters: {json.dumps(tool_selection['parameters'], indent=2)}", "cyan")

            # Execute the selected tool
            try:
                cprint(f"\nInvoking tool: {tool_selection['selected_tool']}", "magenta")
                result = client.tool_runtime.invoke_tool(
                    tool_name=tool_selection['selected_tool'],
                    kwargs=tool_selection['parameters']
                )
                
                # Process and summarize the results
                summary = process_result(client, result, task)
                cprint("\nResults Summary:", "green")
                cprint(summary, "blue")
                
            except Exception as e:
                cprint(f"Error executing tool: {e}", "red")
                import traceback
                cprint(traceback.format_exc(), "red")

    except Exception as e:
        cprint(f"Error: {e}", "red")
        import traceback
        cprint(traceback.format_exc(), "red")

if __name__ == "__main__":
    main()