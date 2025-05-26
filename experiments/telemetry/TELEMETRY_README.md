# Llama Stack Telemetry Guide

Complete guide to using Llama Stack's built-in telemetry features for tracing and analyzing your LLM/agent conversations.

## 🎯 What is Llama Stack Telemetry?

Llama Stack includes a comprehensive telemetry system that automatically traces:
- **Agent conversations** and turns
- **Tool calls** and responses  
- **Model inference** requests and responses
- **Safety checks** and content filtering
- **Vector operations** (embeddings, retrieval)
- **Custom events** and metrics

All telemetry data is captured using OpenTelemetry standards and can be stored locally or exported to external monitoring systems.

## 🔧 Current Configuration

Your telemetry is already configured in your YAML file:

```yaml
telemetry:
- provider_id: meta-reference
  provider_type: inline::meta-reference
  config:
    service_name: llama-stack
    sinks: console,sqlite
    sqlite_db_path: ~/.llama/distributions/distribution-myenv-ollama/trace_store.db
```

### Available Sinks

- **`console`** - Real-time logging to console output
- **`sqlite`** - Persistent storage in SQLite database
- **`otel_trace`** - Export traces to OpenTelemetry collector
- **`otel_metric`** - Export metrics to OpenTelemetry collector

## 📊 What Gets Traced

### Automatic Tracing
Llama Stack automatically creates traces for:

1. **Agent Operations**
   - Agent creation and initialization
   - Session management
   - Turn processing (user message → assistant response)

2. **Model Inference**
   - Prompt preparation and tokenization
   - Model API calls (to Ollama, OpenAI, etc.)
   - Response generation and streaming
   - Token counting and metrics

3. **Tool Execution**
   - Tool discovery and selection
   - Tool parameter validation
   - Tool execution and results
   - Error handling

4. **Safety & Content Filtering**
   - Input safety checks
   - Output safety validation
   - Shield application

5. **Vector Operations**
   - Document embedding
   - Vector similarity search
   - RAG retrieval operations

### Trace Structure

Each trace contains:
- **Trace ID** - Unique identifier for the entire conversation
- **Spans** - Individual operations within the trace
- **Attributes** - Metadata like model names, messages, timing
- **Status** - Success/error status for each operation

## 🛠️ Using the Tools

### 1. Telemetry Guide Script

Run the comprehensive demonstration:

```bash
python telemetry_guide.py
```

This script will:
- Test your telemetry API connection
- Create a sample conversation to generate traces
- Query and analyze the telemetry data
- Show you how to use both the API and local database

### 2. Trace Viewer CLI

Browse your traces with the command-line tool:

```bash
# List recent traces
python trace_viewer.py list

# Show detailed trace information
python trace_viewer.py show <trace_id>

# Search traces by content
python trace_viewer.py search "hello"

# Show database statistics
python trace_viewer.py stats
```

## 📈 Telemetry API Usage

### Query Recent Traces

```python
import llama_stack_client

client = llama_stack_client.LlamaStackClient(base_url="http://localhost:5002")

# Get recent traces
traces = await client.telemetry.query_traces(limit=10)
for trace in traces.data:
    print(f"Trace: {trace.trace_id} ({trace.start_time})")
```

### Get Trace Details

```python
# Get specific trace
trace = await client.telemetry.get_trace(trace_id)

# Get span tree (hierarchical view)
span_tree = await client.telemetry.get_span_tree(
    span_id=trace.root_span_id,
    max_depth=10
)

for span_id, span in span_tree.data.items():
    print(f"Span: {span.name} ({span.start_time} - {span.end_time})")
    if span.attributes:
        print(f"  Attributes: {span.attributes}")
```

### Search and Filter

```python
# Find traces with errors
error_traces = await client.telemetry.query_traces(
    attribute_filters=[
        {"key": "status", "op": "eq", "value": "error"}
    ]
)

# Find spans for specific model
model_spans = await client.telemetry.query_spans(
    attribute_filters=[
        {"key": "model", "op": "eq", "value": "llama3.2:3b-instruct-fp16"}
    ],
    attributes_to_return=["model", "user_message", "duration"]
)
```

## 🗄️ Database Schema

The SQLite database contains two main tables:

### `traces` table
- `trace_id` - Unique trace identifier
- `root_span_id` - ID of the root span
- `start_time` - When the trace started
- `end_time` - When the trace completed

### `spans` table  
- `span_id` - Unique span identifier
- `trace_id` - Parent trace ID
- `parent_span_id` - Parent span (for hierarchy)
- `name` - Operation name
- `start_time` - Span start time
- `end_time` - Span end time
- `attributes` - JSON metadata
- `status` - ok/error status

## 🔍 Analysis Examples

### Find Slow Operations

```sql
SELECT name, AVG(
    (julianday(end_time) - julianday(start_time)) * 86400000
) as avg_duration_ms
FROM spans 
WHERE end_time IS NOT NULL
GROUP BY name 
ORDER BY avg_duration_ms DESC;
```

### Conversation Analysis

```python
from telemetry_guide import TelemetryAnalyzer

analyzer = TelemetryAnalyzer("/path/to/trace_store.db")

# Get recent traces
traces = analyzer.get_recent_traces(limit=5)

# Analyze specific conversation
analysis = analyzer.analyze_conversation_flow(trace_id)
print(f"Total duration: {analysis['timing_analysis']['total_duration_ms']}ms")
print(f"Tool calls: {len(analysis['tool_calls'])}")
```

### Error Tracking

```python
# Find traces with errors
error_traces = await client.telemetry.query_traces(
    attribute_filters=[
        {"key": "error", "op": "eq", "value": "true"}
    ]
)

for trace in error_traces.data:
    # Get error details
    spans = await client.telemetry.query_spans(
        attribute_filters=[
            {"key": "trace_id", "op": "eq", "value": trace.trace_id},
            {"key": "status", "op": "eq", "value": "error"}
        ],
        attributes_to_return=["error_message", "stack_trace"]
    )
```

## 🚀 Advanced Configuration

### Add OpenTelemetry Export

To export traces to external systems like Jaeger:

```yaml
telemetry:
- provider_id: meta-reference
  provider_type: inline::meta-reference
  config:
    service_name: llama-stack
    sinks: console,sqlite,otel_trace
    otel_trace_endpoint: "http://localhost:4318/v1/traces"
    sqlite_db_path: ~/.llama/distributions/distribution-myenv-ollama/trace_store.db
```

### Custom Event Logging

```python
from llama_stack.apis.telemetry import UnstructuredLogEvent, LogSeverity
from datetime import datetime

# Log custom events
event = UnstructuredLogEvent(
    trace_id="your-trace-id",
    span_id="your-span-id", 
    timestamp=datetime.now(),
    message="Custom operation completed",
    severity=LogSeverity.INFO,
    attributes={"custom_metric": 42}
)

await client.telemetry.log_event(event)
```

## 📊 Monitoring Dashboard Ideas

### Key Metrics to Track

1. **Performance Metrics**
   - Average response time per model
   - Token generation rate
   - Tool execution time

2. **Usage Metrics**
   - Conversations per hour/day
   - Most used tools
   - Error rates

3. **Quality Metrics**
   - Safety violations
   - Failed tool calls
   - User satisfaction (if tracked)

### Sample Queries

```python
# Average response time by model
spans = await client.telemetry.query_spans(
    attribute_filters=[
        {"key": "name", "op": "eq", "value": "inference"}
    ],
    attributes_to_return=["model", "duration", "token_count"]
)

# Group by model and calculate averages
model_stats = {}
for span in spans.data:
    model = span.attributes.get("model")
    duration = span.attributes.get("duration")
    if model and duration:
        if model not in model_stats:
            model_stats[model] = []
        model_stats[model].append(duration)

for model, durations in model_stats.items():
    avg_duration = sum(durations) / len(durations)
    print(f"{model}: {avg_duration:.2f}ms average")
```

## 🔧 Troubleshooting

### Common Issues

1. **Database not found**
   - Check the path in your YAML configuration
   - Ensure the directory exists and is writable

2. **No traces appearing**
   - Verify telemetry is enabled in your configuration
   - Check console output for telemetry errors
   - Ensure SQLAlchemy is installed

3. **API connection errors**
   - Verify your Llama Stack server is running
   - Check the base URL in your client

### Debug Commands

```bash
# Check if database exists and has data
python trace_viewer.py stats

# Test telemetry API
python telemetry_guide.py

# Check database schema
sqlite3 ~/.llama/distributions/distribution-myenv-ollama/trace_store.db ".schema"
```

## 🎯 Next Steps

1. **Start your Llama Stack server** with telemetry enabled
2. **Run the telemetry guide** to see it in action
3. **Have some conversations** with your agents
4. **Browse your traces** using the trace viewer
5. **Build custom analysis** for your specific use cases

The telemetry system gives you complete visibility into your LLM/agent operations, helping you optimize performance, debug issues, and understand usage patterns.
