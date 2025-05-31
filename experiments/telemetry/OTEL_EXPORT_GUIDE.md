# OpenTelemetry Export Guide for Llama Stack Conversations

This guide shows you how to export your Llama Stack conversation traces to OpenTelemetry format for import into other observability tools.

## 🚀 Quick Start

**Note**: Run commands from the `experiments/telemetry/` directory:
```bash
cd experiments/telemetry
```

### Export All Conversations to OTLP JSON
```bash
# Export all conversations to OpenTelemetry format
python otel_conversation_exporter.py --output conversations.json

# Export with compression
python otel_conversation_exporter.py --output conversations.json.gz --compress

# Export only recent conversations (last 7 days)
python otel_conversation_exporter.py --output recent_conversations.json --days 7
```

### Export to Jaeger Format
```bash
# Export to Jaeger-compatible JSON format
python otel_conversation_exporter.py --output conversations_jaeger.json --format jaeger

# Export recent conversations to Jaeger format
python otel_conversation_exporter.py --output recent_jaeger.json --format jaeger --days 30
```

## 📊 What Gets Exported

### Conversation Structure
- **Each conversation** becomes an OpenTelemetry trace
- **Each message/tool call** becomes a span within the trace
- **All metadata** is preserved as span attributes
- **Timing information** is converted to OTLP format
- **Parent-child relationships** between spans are maintained

### Span Types
- **User messages** → `SPAN_KIND_CLIENT` spans with `user_message` attributes
- **Assistant responses** → `SPAN_KIND_INTERNAL` spans with `assistant_message` attributes  
- **Tool calls** → `SPAN_KIND_CLIENT` spans with tool name, arguments, and results
- **System messages** → `SPAN_KIND_INTERNAL` spans with system content

### Preserved Data
- Original trace IDs and span IDs (as attributes)
- Message content and metadata
- Tool call details (name, arguments, results)
- Response times and durations
- Error status and codes
- Service information (llama-stack-agent)

## 🔧 Command Reference

### Basic Usage
```bash
python otel_conversation_exporter.py --output <file> [options]
```

### Options
- `--output, -o`: Output file path (required)
- `--format`: Export format (`otlp-json` or `jaeger`, default: `otlp-json`)
- `--days`: Only export traces from last N days
- `--compress`: Compress output with gzip (OTLP JSON only)
- `--db`: Custom database path

### Examples
```bash
# Export everything to OTLP JSON
python otel_conversation_exporter.py -o all_conversations.json

# Export last month to compressed OTLP
python otel_conversation_exporter.py -o monthly.json.gz --days 30 --compress

# Export to Jaeger format for last week
python otel_conversation_exporter.py -o weekly_jaeger.json --format jaeger --days 7

# Use custom database location
python otel_conversation_exporter.py -o export.json --db /path/to/trace_store.db
```

## 📤 Import Into Other Tools

### Jaeger
```bash
# 1. Export to Jaeger format
python otel_conversation_exporter.py -o conversations.json --format jaeger

# 2. Import into Jaeger (example with jaeger-query)
# Follow Jaeger documentation for importing JSON traces
```

### Grafana Tempo
```bash
# 1. Export to OTLP JSON
python otel_conversation_exporter.py -o conversations.json

# 2. Configure Tempo to accept OTLP data
# Use Tempo's OTLP receiver configuration
```

### Custom Analysis Tools
```bash
# Export to OTLP JSON for custom processing
python otel_conversation_exporter.py -o data.json

# The JSON follows standard OTLP format:
# {
#   "resourceSpans": [
#     {
#       "resource": { "attributes": [...] },
#       "scopeSpans": [
#         {
#           "scope": { "name": "...", "version": "..." },
#           "spans": [
#             {
#               "traceId": "...",
#               "spanId": "...",
#               "name": "...",
#               "startTimeUnixNano": "...",
#               "endTimeUnixNano": "...",
#               "attributes": [...],
#               "status": { "code": 1 }
#             }
#           ]
#         }
#       ]
#     }
#   ]
# }
```

## 🔍 Example Output

### Sample OTLP Export Structure
```json
{
  "resourceSpans": [
    {
      "resource": {
        "attributes": [
          {
            "key": "service.name",
            "value": {"stringValue": "llama-stack-agent"}
          },
          {
            "key": "conversation.trace_id", 
            "value": {"stringValue": "a430697567ee2f27e24c45f6631a5920"}
          }
        ]
      },
      "scopeSpans": [
        {
          "scope": {
            "name": "llama-stack-conversation-exporter",
            "version": "1.0.0"
          },
          "spans": [
            {
              "traceId": "a430697567ee2f27e24c45f6631a5920",
              "spanId": "1234567890abcdef",
              "name": "user_message",
              "kind": 3,
              "startTimeUnixNano": "1640995200000000000",
              "endTimeUnixNano": "1640995201000000000",
              "attributes": [
                {
                  "key": "user_message",
                  "value": {"stringValue": "How do I create a web scraper?"}
                },
                {
                  "key": "service.name",
                  "value": {"stringValue": "llama-stack-agent"}
                }
              ],
              "status": {"code": 1}
            }
          ]
        }
      ]
    }
  ]
}
```

### Sample Export Output
```
🔍 Scanning for conversation traces...
📊 Found 12 conversation traces to export
🔄 Processing trace 1/12: a430697567ee2f27...
🔄 Processing trace 2/12: b541708678bf3c38...
...
💾 Writing OTLP export to conversations.json...
✅ Successfully exported 12 traces to conversations.json
📈 Total spans exported: 89

💡 Import Instructions:
   • Jaeger: Use jaeger-query to import OTLP JSON
   • Grafana Tempo: Configure OTLP receiver
   • Custom tools: Parse as standard OTLP JSON format
```

## 🎯 Use Cases

### Observability Integration
- **Import into Jaeger** for distributed tracing visualization
- **Send to Grafana Tempo** for trace storage and analysis
- **Integrate with Datadog/New Relic** for APM monitoring
- **Use with Honeycomb** for debugging and performance analysis

### Data Analysis
- **Export for ML training** - structured conversation data
- **Business intelligence** - conversation patterns and metrics
- **Custom dashboards** - build visualizations with your preferred tools
- **Compliance reporting** - structured audit trails

### Migration and Backup
- **Data portability** - move conversations between systems
- **Long-term archival** - standard format for preservation
- **System migration** - export before upgrading/changing systems
- **Disaster recovery** - backup conversation data

## 🛠️ Advanced Usage

### Batch Processing
```bash
# Export by time periods for large datasets
python otel_conversation_exporter.py -o week1.json --days 7
python otel_conversation_exporter.py -o week2.json --days 14 # (last 14 days)
python otel_conversation_exporter.py -o week3.json --days 21 # (last 21 days)
```

### Automated Exports
```bash
#!/bin/bash
# Daily export script
DATE=$(date +%Y-%m-%d)
python otel_conversation_exporter.py -o "exports/conversations_${DATE}.json" --days 1
```

### Validation
```bash
# Export and validate JSON structure
python otel_conversation_exporter.py -o test.json --days 1
python -m json.tool test.json > /dev/null && echo "Valid JSON" || echo "Invalid JSON"
```

## 🔧 Troubleshooting

### No Traces Found
```
📭 No conversation traces found.
```
**Solutions:**
- Ensure you have conversation data in your telemetry database
- Check the `--days` parameter isn't too restrictive
- Verify the database path is correct

### Export Errors
```
⚠️  Error processing trace <trace_id>: <error_message>
```
**Solutions:**
- Individual trace errors don't stop the export
- Check the error message for specific issues
- Corrupted traces are skipped automatically

### Large File Sizes
For very large exports:
- Use `--compress` flag for OTLP JSON exports
- Export in smaller time chunks with `--days`
- Consider using streaming tools for very large datasets

## 🚀 Next Steps

1. **Start with a small export**: `python otel_conversation_exporter.py -o test.json --days 1`
2. **Validate the output**: Check the JSON structure and content
3. **Import into your target tool**: Follow the tool's OTLP import documentation
4. **Set up regular exports**: Create scripts for ongoing data export
5. **Integrate with your workflow**: Use exported data for analysis and monitoring

The OpenTelemetry exporter gives you complete data portability for your Llama Stack conversations, enabling integration with the entire observability ecosystem!
