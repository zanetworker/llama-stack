# Llama Stack Telemetry Analysis Tools

This directory contains a comprehensive suite of tools for analyzing Llama Stack telemetry data, enabling you to replay conversations, analyze patterns, and gain insights from your LLM interactions.

## 📁 Files Overview

### 🎬 Analysis Tools
- **`conversation_replay.py`** - Interactive step-by-step conversation replay
- **`message_history_analyzer.py`** - Extract and export full conversation histories  
- **`conversation_patterns.py`** - Analyze patterns and generate insights
- **`trace_viewer.py`** - General trace inspection and debugging

### 📚 Documentation
- **`CONVERSATION_ANALYSIS_GUIDE.md`** - Complete usage guide with examples
- **`TELEMETRY_README.md`** - General telemetry setup and configuration
- **`telemetry_working_demo.py`** - Working demo showing telemetry integration

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

## 📖 Full Documentation

See **`CONVERSATION_ANALYSIS_GUIDE.md`** for:
- Detailed usage instructions
- Example workflows
- Sample outputs
- Troubleshooting guide
- Advanced usage patterns

## 🎯 What These Tools Enable

- **🎬 Replay entire conversations** step-by-step with timing and tool details
- **📊 Analyze usage patterns** - peak hours, user behavior, system performance
- **📤 Export conversation data** in multiple formats (JSON, CSV, Markdown)
- **🔍 Debug performance issues** by identifying slow operations and errors
- **📈 Generate insights** for improving your LLM applications

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

### For Product Managers  
- Track user engagement and feature usage
- Monitor system health and reliability
- Generate reports for stakeholders

### For Data Scientists
- Export structured conversation data
- Analyze patterns for model improvements
- Study user behavior across time periods

---

**Get started**: Open `CONVERSATION_ANALYSIS_GUIDE.md` for complete instructions and examples!
