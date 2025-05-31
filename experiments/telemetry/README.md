# Llama Stack Telemetry Analysis Tools

This directory contains a comprehensive suite of tools for analyzing Llama Stack telemetry data, enabling you to replay conversations, analyze patterns, and gain insights from your LLM interactions.

## 📁 Files Overview

### 🎬 Analysis Tools
- **`conversation_replay.py`** - Interactive step-by-step conversation replay
- **`message_history_analyzer.py`** - Extract and export full conversation histories  
- **`conversation_patterns.py`** - Analyze patterns and generate insights
- **`trace_viewer.py`** - General trace inspection and debugging
- **`otel_conversation_exporter.py`** - Export conversations to OpenTelemetry format
- **`otel_validator.py`** - Validate OTLP exports against OpenTelemetry specifications

### 📚 Documentation
- **`CONVERSATION_ANALYSIS_GUIDE.md`** - Complete usage guide with examples
- **`OTEL_EXPORT_GUIDE.md`** - OpenTelemetry export instructions and examples
- **`OTEL_EXPORTER_USAGE.md`** - Enhanced OTEL exporter usage guide with --limit examples
- **`OTEL_VALIDATION_GUIDE.md`** - OTLP validation and conformance testing
- **`CONVERSATION_DATA_SUMMARY.md`** - Summary of available conversation data

## 🚀 Quick Start

1. **Navigate to this directory**:
   ```bash
   cd experiments/telemetry
   ```

2. **Start with a comprehensive analysis**:
   ```bash
   python conversation_patterns.py report
   ```

3. **List and replay conversations**:
   ```bash
   python conversation_replay.py list
   python conversation_replay.py replay <trace_id>
   ```

4. **Export data for analysis**:
   ```bash
   python message_history_analyzer.py export --format json
   ```

5. **Export to OpenTelemetry format**:
   ```bash
   # Export limited conversations for testing
   python otel_conversation_exporter.py -o test.json --limit 2
   
   # Export recent conversations
   python otel_conversation_exporter.py -o conversations.json --days 7
   
   # Validate the export
   python otel_validator.py conversations.json
   ```

## 📖 Full Documentation

See **`CONVERSATION_ANALYSIS_GUIDE.md`** for:
- Detailed usage instructions
- Example workflows
- Sample outputs
- Troubleshooting guide
- Advanced usage patterns

See **`OTEL_EXPORT_GUIDE.md`** and **`OTEL_VALIDATION_GUIDE.md`** for:
- OpenTelemetry export instructions
- OTLP format validation
- Integration with observability tools
- Import procedures for Jaeger, Grafana Tempo, etc.

## 🎯 What These Tools Enable

- **🎬 Replay entire conversations** step-by-step with timing and tool details
- **📊 Analyze usage patterns** - peak hours, user behavior, system performance
- **📤 Export conversation data** in multiple formats (JSON, CSV, Markdown, OTLP)
- **🔍 Debug performance issues** by identifying slow operations and errors
- **📈 Generate insights** for improving your LLM applications
- **🔗 Integrate with observability tools** via OpenTelemetry standard format
- **✅ Validate exports** to ensure compatibility with OTLP tools

## 🔧 Requirements

These tools work with Llama Stack's built-in telemetry system. Make sure you have:
- Telemetry enabled in your Llama Stack configuration
- SQLite database with trace data (default location is automatically detected)
- Python 3.7+ with standard libraries

## 💡 Use Cases

### For Developers
- Debug conversation flows and identify issues
- Understand user interaction patterns
- Optimize response times and reduce errors
- Export traces to observability platforms

### For Product Managers  
- Track user engagement and feature usage
- Monitor system health and reliability
- Generate reports for stakeholders

### For Data Scientists
- Export structured conversation data
- Analyze patterns for model improvements
- Study user behavior across time periods

### For DevOps/SRE
- Integrate with existing observability stack
- Monitor LLM application performance
- Set up alerts and dashboards
- Ensure data quality with validation

---

**Get started**: Open `CONVERSATION_ANALYSIS_GUIDE.md` for complete instructions and examples!
