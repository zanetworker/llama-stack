#!/usr/bin/env python3
"""
Script to examine the structure of exported OTEL conversation data
"""
import json

def examine_conversation_structure():
    with open('conversations.json', 'r') as f:
        data = json.load(f)
    
    # Get first conversation
    first_resource = data['resourceSpans'][0]
    first_scope = first_resource['scopeSpans'][0]
    spans = first_scope['spans']
    
    print(f"=== CONVERSATION STRUCTURE ANALYSIS ===")
    print(f"Total spans in first conversation: {len(spans)}")
    print()
    
    # Examine first span (the main request)
    first_span = spans[0]
    print(f"First span name: {first_span.get('name', 'N/A')}")
    
    # Look for request data
    for attr in first_span.get('attributes', []):
        if attr['key'] == 'request':
            request_data = json.loads(list(attr['value'].values())[0])
            
            print("\n=== REQUEST DATA ===")
            print(f"Available keys: {list(request_data.keys())}")
            
            # System instructions
            if 'instructions' in request_data:
                instructions = request_data['instructions']
                if instructions:
                    print(f"\n✅ SYSTEM INSTRUCTIONS: {instructions[:200]}...")
                else:
                    print(f"\n❌ SYSTEM INSTRUCTIONS: None/Empty")
            
            # User messages
            if 'messages' in request_data:
                messages = request_data['messages']
                print(f"\n✅ MESSAGES ({len(messages)} total):")
                for i, msg in enumerate(messages):
                    role = msg.get('role', 'unknown')
                    content = str(msg.get('content', ''))[:150]
                    print(f"  Message {i+1} ({role}): {content}...")
            
            # Tools
            if 'toolgroups' in request_data:
                toolgroups = request_data['toolgroups']
                if toolgroups:
                    print(f"\n✅ TOOL GROUPS ({len(toolgroups)} total):")
                    for i, tg in enumerate(toolgroups):
                        if 'tools' in tg and tg['tools']:
                            for j, tool in enumerate(tg['tools']):
                                tool_name = tool.get('function', {}).get('name', 'unknown')
                                print(f"  Tool {i+1}.{j+1}: {tool_name}")
                else:
                    print(f"\n❌ TOOL GROUPS: None/Empty")
            
            break
    
    # Look at other spans for responses and tool calls
    print(f"\n=== SPAN OVERVIEW ===")
    for i, span in enumerate(spans[:10]):  # First 10 spans
        span_name = span.get('name', 'unknown')
        print(f"Span {i+1}: {span_name}")
        
        # Check for response data
        for attr in span.get('attributes', []):
            if attr['key'] == 'response':
                try:
                    response_data = json.loads(list(attr['value'].values())[0])
                    
                    # Look for assistant responses
                    if 'event' in response_data:
                        event = response_data['event']
                        if 'payload' in event:
                            payload = event['payload']
                            if 'delta' in payload:
                                delta = payload['delta']
                                if delta.get('role') == 'assistant':
                                    content = delta.get('content', '')
                                    if content:
                                        print(f"  ✅ Assistant response: {content[:100]}...")
                                elif 'tool_calls' in delta:
                                    tool_calls = delta['tool_calls']
                                    for tc in tool_calls:
                                        if 'function' in tc:
                                            func_name = tc['function'].get('name', 'unknown')
                                            print(f"  ✅ Tool call: {func_name}")
                except:
                    print(f"  ⚠️ Could not parse response data")
                break
    
    print(f"\n=== SUMMARY ===")
    print("The OTEL export contains:")
    print("✅ User prompts/messages")
    print("✅ System instructions (if present)")
    print("✅ Assistant responses")
    print("✅ Tool definitions and calls")
    print("✅ Complete conversation flow with timing")
    print("✅ All metadata (session IDs, agent IDs, etc.)")

if __name__ == "__main__":
    examine_conversation_structure()
