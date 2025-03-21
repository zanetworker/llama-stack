from llama_stack_client import LlamaStackClient
from termcolor import cprint
import os
import json

# Initialize the client
client = LlamaStackClient(
    base_url=f"http://localhost:{os.environ.get('LLAMA_STACK_PORT', '8080')}"
)

# Check available providers
available_providers = client.providers.list()
cprint(f"Available providers: {[p.provider_id for p in available_providers]}", "yellow")

# Find the MCP provider
mcp_provider = next((p for p in available_providers if p.provider_id == "model-context-protocol"), None)
if not mcp_provider:
    cprint("MCP provider not found. Please make sure it's installed and enabled.", "red")
    exit(1)

cprint(f"Using MCP provider: {mcp_provider.provider_id}", "green")

# Try to unregister the toolgroup if it exists
try:
    cprint("Unregistering existing toolgroup", "yellow")
    client.toolgroups.unregister(toolgroup_id="mcp::docling")
    cprint("Unregistered existing toolgroup", "yellow")
except Exception:
    pass  # Ignore if it doesn't exist

# Register MCP tools
try:
    cprint("Registering MCP docling toolgroup", "yellow")
    client.toolgroups.register(
        toolgroup_id="mcp::docling",
        provider_id="model-context-protocol",
        mcp_endpoint={"uri": "http://0.0.0.0:8000/sse"})
    cprint("Successfully registered MCP docling toolgroup", "green")
except Exception as e:
    cprint(f"Error registering MCP toolgroup: {e}", "red")
    exit(1)

# List available tools in the toolgroup
cprint("Listing tools in the toolgroup...", "yellow")
tools = client.tools.list(toolgroup_id="mcp::docling")
cprint(f"Available tools: {[t.identifier for t in tools]}", "cyan")

# Get available tools
if tools:
    for tool in tools:
        cprint(f"\nTool: {tool.identifier}", "cyan")
        cprint(f"Description: {tool.description}", "cyan")

        # Display parameters with their details
        cprint("\nParameters:", "cyan")
        for param in tool.parameters:
            param_type = getattr(param, 'parameter_type', 'unknown')
            param_desc = getattr(param, 'description', 'No description')
            param_required = getattr(param, 'required', False)
            cprint(f"  - {param.name} ({param_type}): {param_desc} {'(Required)' if param_required else ''}", "cyan")

        # Example invocation for document conversion
        if tool.identifier == "convert_document":
            try:
                cprint(f"\nInvoking tool: {tool.identifier}", "magenta")
                result = client.tool_runtime.invoke_tool(
                    tool_name=tool.identifier,
                    kwargs={
                        "source": "https://example.com/sample.pdf",
                        "format": "markdown"
                    }
                )
                cprint("\nTool invocation result:", "green")
                cprint(str(result)[:200] + "...", "blue")

            except Exception as e:
                cprint(f"Error invoking tool: {e}", "red")
                import traceback
                cprint(traceback.format_exc(), "red")

        # Example invocation for table extraction
        elif tool.identifier == "extract_tables":
            try:
                cprint(f"\nInvoking tool: {tool.identifier}", "magenta")
                result = client.tool_runtime.invoke_tool(
                    tool_name=tool.identifier,
                    kwargs={
                        "source": "https://example.com/sample.pdf"
                    }
                )
                cprint("\nTool invocation result:", "green")
                cprint(str(result)[:200] + "...", "blue")

            except Exception as e:
                cprint(f"Error invoking tool: {e}", "red")
                import traceback
                cprint(traceback.format_exc(), "red")

        # Process results with LLM if needed
        try:
            model_id = os.environ["INFERENCE_MODEL"]
            cprint(f"Using model {model_id} for processing", "green")
            
            prompt = f"""
            I have processed a document using MCP tools. Please analyze and summarize the results.
            
            Results:
            {str(result)[:5000]}  # Limit to first 5000 chars
            """

            response = client.inference.chat_completion(
                model_id=model_id,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes document processing results."},
                    {"role": "user", "content": prompt}
                ]
            )

            if hasattr(response, 'completion_message') and hasattr(response.completion_message, 'content'):
                cprint("\nAnalysis:", "green")
                cprint(response.completion_message.content, "blue")

        except Exception as e:
            cprint(f"Error processing with LLM: {e}", "red")
            import traceback
            cprint(traceback.format_exc(), "red")

else:
    cprint("No tools found in the toolgroup", "red")