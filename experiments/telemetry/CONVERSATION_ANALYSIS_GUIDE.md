# Llama Stack Conversation Analysis Guide

This guide shows you how to replay entire conversations, analyze message histories, and discover patterns in your Llama Stack telemetry data.

## 🎬 Tools Overview

You now have four powerful tools for conversation analysis:

1. **`conversation_replay.py`** - Interactive step-by-step conversation replay
2. **`message_history_analyzer.py`** - Extract and export full conversation histories
3. **`conversation_patterns.py`** - Analyze patterns and generate insights
4. **`trace_viewer.py`** - General trace inspection (existing tool)

## 🚀 Quick Start

**Note**: All commands should be run from the `experiments/telemetry/` directory:
```bash
cd experiments/telemetry
```

### 1. List Available Conversations
```bash
# See what conversations are available
python conversation_replay.py list

# Limit to last 10 conversations
python conversation_replay.py list --limit 10
```

### 2. Replay a Conversation Step-by-Step
```bash
# Interactive replay (press Enter for each turn)
python conversation_replay.py replay a430697567ee2f27e24c45f6631a5920

# Show all at once without pausing
python conversation_replay.py replay a430697567ee2f27e24c45f6631a5920 --no-interactive
```

### 3. Analyze Conversation Patterns
```bash
# Generate comprehensive analysis report
python conversation_patterns.py report

# Analyze last 7 days only
python conversation_patterns.py report --days 7

# Get specific analysis types
python conversation_patterns.py usage --days 30
python conversation_patterns.py performance --days 30
python conversation_patterns.py content --days 30
```

### 4. Export Conversation Data
```bash
# Export to JSON
python message_history_analyzer.py export --format json

# Export to CSV for spreadsheet analysis
python message_history_analyzer.py export --format csv

# Export to Markdown for documentation
python message_history_analyzer.py export --format markdown

# Analyze and export in one command
python message_history_analyzer.py analyze --export json
```

## 📊 What You Can Discover

### 🎬 Conversation Replay Features
- **Step-by-step playback** of entire conversations
- **Color-coded messages** (user, assistant, tool, system)
- **Response timing** for each turn
- **Tool call details** and results
- **Conversation summaries** with statistics

### 📈 Pattern Analysis Insights
- **Usage patterns**: Peak hours, busiest days, activity trends
- **Performance analysis**: Slowest operations, error rates, bottlenecks
- **Content patterns**: Message lengths, common phrases, tool usage
- **User engagement**: Turn counts, conversation lengths, interaction styles

### 📤 Export Capabilities
- **JSON format**: Complete structured data for programmatic analysis
- **CSV format**: Spreadsheet-friendly for business analysis
- **Markdown format**: Human-readable reports for documentation

## 🔍 Example Workflows

### Workflow 1: Debug a Slow Conversation
```bash
# 1. Find conversations with performance issues
python conversation_patterns.py performance --days 7

# 2. Identify the slowest trace ID from the output
python conversation_replay.py replay <slow_trace_id>

# 3. Step through to see where delays occur
# Look for long response times between turns
```

### Workflow 2: Understand User Behavior
```bash
# 1. Get comprehensive analysis
python conversation_patterns.py report --days 30

# 2. Export detailed data for further analysis
python message_history_analyzer.py export --format csv --days 30

# 3. Open the CSV in Excel/Google Sheets for custom analysis
```

### Workflow 3: Monitor System Health
```bash
# Daily health check
python conversation_patterns.py report --days 1

# Weekly trend analysis
python conversation_patterns.py usage --days 7

# Monthly performance review
python conversation_patterns.py performance --days 30
```

### Workflow 4: Content Analysis for Improvements
```bash
# Analyze what users are asking about
python conversation_patterns.py content --days 30

# Export conversations for manual review
python message_history_analyzer.py export --format markdown --days 7

# Look for common patterns to improve responses
```

## 📋 Sample Output Examples

### Conversation Replay Output
```
🎬 Replaying Conversation: a430697567ee2f27e24c45f6631a5920
============================================================

--- Turn 1 (15:49:37) ---

👤 User:
How do I create a simple web scraper in Python?

🤖 Assistant:
I'll help you create a simple web scraper in Python using the requests and BeautifulSoup libraries...

⏱️  Response time: 1250ms

--- Turn 2 (15:50:15) ---

👤 User:
Can you show me how to handle errors in the scraper?

🔧 Tool:
Tool: web_search
Arguments: {"query": "python web scraping error handling best practices"}
Result: Found 5 relevant articles about error handling...

🤖 Assistant:
Great question! Here are the key error handling techniques for web scrapers...

⏱️  Response time: 2100ms
```

### Pattern Analysis Report
```
🔍 Comprehensive Conversation Analysis Report
📅 Analyzing data from the last 30 days
======================================================================

📊 Usage Patterns
------------------------------
Total conversations: 45
Average daily usage: 1.5
Peak hour: 15:00 (12 conversations)
Peak day: Wednesday (18 conversations)

⚡ Performance Analysis
------------------------------
Total operations analyzed: 156
Overall error rate: 2.1%

🐌 Slowest Operations:
  /v1/inference/chat_completion: 1847ms avg (23 calls)
  /v1/agents/turn: 1205ms avg (45 calls)

💬 Content Analysis
------------------------------
User messages: 89
Assistant messages: 91
Average message length: 247 characters

🔧 Tool Usage:
  web_search: 12 times
  code_interpreter: 8 times

💡 Key Insights & Recommendations
----------------------------------------
📈 System processed 45 conversations in the analyzed period
⏰ Peak usage occurs at 15:00 with 12 conversations
🐌 Slowest operation: '/v1/inference/chat_completion' averaging 1847ms
💬 Assistant responds 1.0 times per user message on average

🎯 Recommendations:
• Consider optimizing the slowest operations for better user experience
• Tool usage indicates users are engaging with advanced features
```

## 🛠️ Advanced Usage

### Custom Database Path
All tools support custom database paths:
```bash
python conversation_replay.py --db /path/to/your/trace_store.db list
```

### Filtering by Time Period
Most tools support time-based filtering:
```bash
# Last 24 hours
python conversation_patterns.py report --days 1

# Last week
python message_history_analyzer.py analyze --days 7

# Last month
python conversation_replay.py list --limit 50  # (shows recent conversations)
```

### Combining Tools
Chain tools together for comprehensive analysis:
```bash
# 1. Get overview
python conversation_patterns.py report --days 7

# 2. Export data for detailed analysis
python message_history_analyzer.py export --format json --days 7

# 3. Replay interesting conversations
python conversation_replay.py list --limit 10
python conversation_replay.py replay <interesting_trace_id>
```

## 🎯 What This Enables

### For Developers
- **Debug conversation flows** by replaying them step-by-step
- **Identify performance bottlenecks** in your LLM pipeline
- **Understand user interaction patterns** to improve UX
- **Export data** for custom analysis and reporting

### For Product Managers
- **Track user engagement** and conversation quality
- **Identify popular features** through tool usage analysis
- **Monitor system health** and error rates
- **Generate reports** for stakeholders

### For Data Scientists
- **Analyze conversation patterns** for insights
- **Export structured data** for ML model training
- **Study user behavior** across different time periods
- **Identify optimization opportunities** in the system

## 🔧 Troubleshooting

### No Conversations Found
If you see "No conversation traces found":
1. Make sure your Llama Stack server is running and capturing telemetry
2. Have some conversations with agents to generate data
3. Check that the database path is correct
4. Verify the time period isn't too restrictive

### Database Errors
If you get database connection errors:
1. Ensure the database file exists at the specified path
2. Check file permissions
3. Verify the database isn't locked by another process

### Empty Analysis Results
If analysis shows no data:
1. Increase the `--days` parameter to look further back
2. Check that conversations contain the expected message attributes
3. Verify that telemetry is properly configured and capturing data

## 🚀 Next Steps

1. **Start with the comprehensive report**: `python conversation_patterns.py report`
2. **Replay your most recent conversation**: Use `conversation_replay.py list` then replay one
3. **Export data for deeper analysis**: Try different export formats
4. **Set up regular monitoring**: Create scripts to run these tools daily/weekly
5. **Customize for your needs**: Modify the tools to add specific analysis for your use case

These tools give you complete visibility into your Llama Stack conversations, enabling data-driven improvements to your LLM applications!
