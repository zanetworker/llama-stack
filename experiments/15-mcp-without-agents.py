from llama_stack_client import LlamaStackClient
from llama_stack_client.types.shared_params.url import URL
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
    client.toolgroups.unregister(toolgroup_id="mcp::filesystem")
    cprint("Unregistered existing toolgroup", "yellow")
except Exception:
    pass  # Ignore if it doesn't exist

# Register MCP tools
try:
    client.toolgroups.register(
        toolgroup_id="mcp::filesystem",
        provider_id="model-context-protocol",
        mcp_endpoint=URL(uri="http://localhost:8000/sse"))
    cprint("Successfully registered MCP toolgroup", "green")
except Exception as e:
    cprint(f"Error registering MCP toolgroup: {e}", "red")
    exit(1)

# List available tools in the toolgroup
tools = client.tools.list(toolgroup_id="mcp::filesystem")
cprint(f"Available tools: {[t.identifier for t in tools]}", "cyan")

# Get the first tool
if tools:
    tool = tools[0]
    cprint(f"\nTool: {tool.identifier}", "cyan")
    cprint(f"Description: {tool.description}", "cyan")
    
    # Display parameters with their details
    cprint("\nParameters:", "cyan")
    for param in tool.parameters:
        param_type = getattr(param, 'parameter_type', 'unknown')
        param_desc = getattr(param, 'description', 'No description')
        param_required = getattr(param, 'required', False)
        cprint(f"  - {param.name} ({param_type}): {param_desc} {'(Required)' if param_required else ''}", "cyan")
    
    # Create kwargs dynamically based on the first parameter
    # In a real application, you would prompt the user for each required parameter
    kwargs = {}
    if tool.parameters:
        first_param = tool.parameters[0]
        # For demonstration, we're using a URL for any parameter, but in a real app
        # you would use appropriate values based on parameter type
        kwargs[first_param.name] = "https://github.com/DS4SD/docling-jobkit"
        cprint(f"\nUsing parameter {first_param.name}={kwargs[first_param.name]}", "yellow")
    
    # Direct tool invocation
    try:
        cprint(f"\nInvoking tool: {tool.identifier} with {kwargs}", "magenta")
        result = client.tool_runtime.invoke_tool(
            tool_name=tool.identifier,
            kwargs=kwargs
        )
        
        # Print the raw result
        cprint("\nTool invocation result:", "green")
        if hasattr(result, 'content'):
            content = result.content
            if isinstance(content, dict) and 'text' in content:
                raw_content = content['text']
                cprint(f"Raw content length: {len(raw_content)} characters", "blue")
                cprint(f"Raw content preview: {raw_content[:200]}...", "blue")
            else:
                raw_content = str(content)
                cprint(f"Raw content length: {len(raw_content)} characters", "blue")
                cprint(f"Raw content preview: {raw_content[:200]}...", "blue")
        else:
            raw_content = str(result)
            cprint(f"Raw result: {raw_content[:200]}...", "blue")
        
        # Format the content using an LLM
        try:
            # Get available models
            models = client.models.list()
            cprint(f"Available models: {[m.identifier for m in models]}", "yellow")
            
            # Choose the first available model
            if models:
                model_id = models[1].identifier
                cprint(f"Using model {model_id} for formatting", "green")
                
                # Prepare the prompt
                prompt = f"""
                I have fetched content from a website. Please format and summarize this content in a readable way.
                Focus on the main elements of the page and provide a structured summary.
                
                Raw content:
                {raw_content[:5000]}  # Limit to first 5000 chars to avoid token limits
                """
                
                # Call the model using chat_completion
                cprint("\nFormatting content with LLM...", "magenta")
                response = client.inference.chat_completion(
                    model_id=model_id,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that summarizes web content."},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                # Display the formatted result
                cprint("\nFormatted content:", "green")
                if hasattr(response, 'completion_message') and hasattr(response.completion_message, 'content'):
                    cprint(response.completion_message.content, "blue")
                else:
                    cprint(f"Unexpected response format: {response}", "red")
            else:
                cprint("No models available for formatting", "red")
        except Exception as e:
            cprint(f"Error formatting with LLM: {e}", "red")
            import traceback
            cprint(traceback.format_exc(), "red")
            
    except Exception as e:
        cprint(f"Error invoking tool: {e}", "red")
        import traceback
        cprint(traceback.format_exc(), "red")
else:
    cprint("No tools found in the toolgroup", "red")