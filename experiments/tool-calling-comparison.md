# Tool Calling: OpenAI vs LlamaStack Comparison

This document compares the implementation differences between OpenAI and LlamaStack for tool calling functionality.

## 1. Basic Request Structure

### OpenAI
```python
{
    'model': 'gpt-3.5-turbo',
    'messages': [...],
    'tools': [...],
    'tool_choice': 'auto',
    'response_format': { 'type': 'json_object' }
}
```

### LlamaStack
```python
{
    'model_id': 'llama3.2:3b-instruct-fp16',
    'messages': [...],
    'tools': [...],
    'tool_choice': 'auto',
    'tool_prompt_format': 'python_list',  # LlamaStack specific
    'response_format': {
        'type': 'json_schema',
        'json_schema': {...}
    }
}
```

Key differences:
- LlamaStack uses `model_id` instead of `model`
- LlamaStack adds `tool_prompt_format` parameter
- Response format uses more detailed JSON schema in LlamaStack

## 2. Tool Definition Structure

### OpenAI
```python
{
    'tools': [{
        'type': 'function',
        'function': {
            'name': 'get_weather',
            'description': 'Get current temperature for provided coordinates in celsius.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'latitude': {
                        'type': 'number',
                        'description': 'latitude of city'
                    },
                    'longitude': {
                        'type': 'number',
                        'description': 'longitude of city'
                    }
                },
                'required': ['latitude', 'longitude']
            }
        }
    }]
}
```

### LlamaStack
```python
{
    'tools': [{
        'tool_name': 'get_weather',
        'description': 'Get current temperature for provided coordinates in celsius.',
        'parameters': {
            'latitude': {
                'param_type': 'float',
                'description': 'latitude of city'
            },
            'longitude': {
                'param_type': 'float',
                'description': 'longitude of city'
            }
        }
    }]
}
```

Key differences:
- OpenAI uses nested `function` object with `type`
- LlamaStack uses flatter structure with `tool_name`
- Parameter types use different naming: `type` vs `param_type`
- OpenAI requires explicit `required` field

## 3. Tool Call Response Structure

### OpenAI
```python
{
    'role': 'assistant',
    'content': None,
    'tool_calls': [{
        'id': 'call_HhxUK3KrjR4Zn8YIX3kNCKs5',
        'type': 'function',
        'function': {
            'name': 'get_weather',
            'arguments': '{"latitude": 33.7489, "longitude": -84.3879}'
        }
    }]
}
```

### LlamaStack
```python
{
    'role': 'assistant',
    'content': '',
    'stop_reason': 'end_of_turn',  # LlamaStack specific
    'tool_calls': [{
        'call_id': '8d1b1822-07c8-437e-b2eb-dcb5089e624a',
        'tool_name': 'get_weather',
        'arguments': {
            'latitude': 33.7489,
            'longitude': -84.3879
        }
    }]
}
```

Key differences:
- LlamaStack includes `stop_reason`
- OpenAI uses string for arguments, LlamaStack uses parsed object
- Different ID field names: `id` vs `call_id`
- OpenAI includes `type` field

## 4. Tool Result Structure

### OpenAI
```python
{
    'role': 'tool',
    'tool_call_id': 'call_HhxUK3KrjR4Zn8YIX3kNCKs5',
    'name': 'get_weather',
    'content': '{"temperature": 22.5, "conditions": "sunny"}'
}
```

### LlamaStack
```python
{
    'role': 'tool',
    'call_id': '8d1b1822-07c8-437e-b2eb-dcb5089e624a',
    'tool_name': 'get_weather',
    'content': '{"time": "2025-02-17T11:15", "temperature_2m": -1.0, "wind_speed_10m": 18.7}'
}
```

Key differences:
- Different ID field names: `tool_call_id` vs `call_id`
- Different tool name fields: `name` vs `tool_name`
- Content structure may vary based on tool implementation

## 5. Response Format

### OpenAI
```python
{
    'response_format': { 'type': 'json_object' }
}
```

### LlamaStack
```python
{
    'response_format': {
        'type': 'json_schema',
        'json_schema': {
            'properties': {
                'temperature': {
                    'description': 'The current temperature in celsius for the given location.',
                    'title': 'Temperature',
                    'type': 'number'
                },
                'response': {
                    'description': "A natural language response to the user's question.",
                    'title': 'Response',
                    'type': 'string'
                }
            },
            'required': ['temperature', 'response'],
            'title': 'WeatherResponse',
            'type': 'object'
        }
    }
}
```

Key differences:
- LlamaStack uses more detailed JSON schema definition
- OpenAI uses simpler format specification
- LlamaStack allows for more structured and validated responses

## 6. Client Usage Example

### OpenAI
```python
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "What's the weather in Atlanta?"}
    ],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"}
                }
            }
        }
    }]
)
```

### LlamaStack
```python
from llama_stack_client import LlamaStackClient
client = LlamaStackClient()
response = client.inference.chat_completion(
    model_id="llama3.2:3b-instruct-fp16",
    messages=[
        {"role": "user", "content": "What's the weather in Atlanta?"}
    ],
    tools=[{
        "tool_name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "latitude": {"param_type": "float"},
            "longitude": {"param_type": "float"}
        }
    }],
    tool_prompt_format="python_list"
)
```

Key differences:
- Different client initialization
- LlamaStack uses `inference.chat_completion`
- LlamaStack requires `tool_prompt_format`
- Simpler tool definition structure in LlamaStack 