# OpenTelemetry Conversation Exporter - Usage Guide

The `otel_conversation_exporter.py` tool exports Llama Stack conversation traces to OpenTelemetry format for integration with observability tools.

## 🚀 Quick Start

```bash
cd experiments/telemetry

# Export just 1 conversation for testing
python otel_conversation_exporter.py -o test.json --limit 1

# Export 2 conversations in Jaeger format
python otel_conversation_exporter.py -o test_jaeger.json --format jaeger --limit 2
```

## 📋 Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--output`, `-o` | Output file path (required) | `-o conversations.json` |
| `--format` | Export format: `otlp-json` (default) or `jaeger` | `--format jaeger` |
| `--days` | Only export traces from last N days | `--days 7` |
| `--limit` | **NEW!** Limit number of conversations (perfect for testing) | `--limit 5` |
| `--compress` | Compress output with gzip (OTLP only) | `--compress` |
| `--db` | Custom database path (auto-detected by default) | `--db /path/to/trace_store.db` |

## 🎯 Common Use Cases

### **Testing & Development**
```bash
# Export just 1 conversation for quick testing
python otel_conversation_exporter.py -o minimal.json --limit 1

# Export 3 conversations for development
python otel_conversation_exporter.py -o dev_sample.json --limit 3
```

### **Recent Activity Analysis**
```bash
# Last 2 conversations from today
python otel_conversation_exporter.py -o today.json --days 1 --limit 2

# Last 5 conversations from this week
python otel_conversation_exporter.py -o week.json --days 7 --limit 5
```

### **Production Exports**
```bash
# All conversations from last 30 days (compressed)
python otel_conversation_exporter.py -o monthly.json.gz --days 30 --compress

# All conversations in Jaeger format
python otel_conversation_exporter.py -o all_jaeger.json --format jaeger
```

## 📊 Export Results

### **File Size Comparison**
- **Without `--limit`**: 17MB+ (all conversations)
- **With `--limit 2`**: ~196KB (manageable for testing)
- **With `--limit 1`**: ~193KB (minimal test data)

### **What Gets Exported**
- ✅ Complete conversation traces
- ✅ User messages and AI responses
- ✅ Tool calls and results
- ✅ System instructions
- ✅ Performance timing data
- ✅ All metadata and attributes

## 🔧 Integration Examples

### **Import into Jaeger**
```bash
# Export in Jaeger format
python otel_conversation_exporter.py -o traces.json --format jaeger --limit 10

# Import into Jaeger (example)
curl -X POST http://localhost:14268/api/traces \
  -H "Content-Type: application/json" \
  -d @traces.json
```

### **Import into Grafana Tempo**
```bash
# Export in OTLP format
python otel_conversation_exporter.py -o traces.json --limit 5

# Configure Tempo to accept OTLP JSON imports
```

## 💡 Pro Tips

1. **Start Small**: Always use `--limit` for initial testing
2. **Combine Filters**: Use `--days` + `--limit` for precise control
3. **Compression**: Use `--compress` for large exports
4. **Format Choice**: Use `jaeger` format for Jaeger imports, `otlp-json` for everything else

## 🔍 Troubleshooting

### **Large Export Files**
```bash
# Problem: Export too large
python otel_conversation_exporter.py -o huge.json --days 30

# Solution: Add limit
python otel_conversation_exporter.py -o manageable.json --days 30 --limit 10
```

### **Testing New Features**
```bash
# Always start with minimal exports
python otel_conversation_exporter.py -o test.json --limit 1
```

## 📈 Performance

- **1 conversation**: ~9 spans, ~193KB
- **2 conversations**: ~18 spans, ~196KB  
- **Typical conversation**: 8-12 spans per conversation
- **Export speed**: ~1-2 conversations per second

The `--limit` parameter is perfect for development, testing, and creating manageable sample datasets!
