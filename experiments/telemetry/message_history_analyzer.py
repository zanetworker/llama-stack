#!/usr/bin/env python3
"""
Message History Analyzer for Llama Stack Telemetry

This tool extracts and analyzes full message histories from telemetry data,
providing detailed conversation analysis and export capabilities.
"""

import argparse
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, asdict
import csv


@dataclass
class ConversationSession:
    """Represents a complete conversation session."""
    session_id: Optional[str]
    agent_id: Optional[str]
    trace_ids: List[str]
    start_time: str
    end_time: str
    total_turns: int
    total_messages: int
    total_tool_calls: int
    duration_ms: float
    user_messages: List[str]
    assistant_messages: List[str]
    topics: Set[str]
    errors: List[str]


@dataclass
class ConversationAnalytics:
    """Analytics data for conversation patterns."""
    total_conversations: int
    total_messages: int
    avg_conversation_length: float
    avg_response_time_ms: float
    most_common_topics: List[tuple]
    peak_usage_hours: List[int]
    error_rate: float
    user_engagement_patterns: Dict[str, Any]


class MessageHistoryAnalyzer:
    """Analyzes message histories and conversation patterns."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def get_all_conversations(self, days_back: int = 30) -> List[ConversationSession]:
        """Extract all conversations from the last N days."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Calculate date threshold
        cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
        
        # Get all traces with conversation data
        query = """
        SELECT DISTINCT t.trace_id, t.root_span_id, t.start_time, t.end_time
        FROM traces t
        JOIN spans s ON t.trace_id = s.trace_id
        WHERE t.start_time > ?
          AND (s.attributes LIKE '%user_message%' 
               OR s.attributes LIKE '%assistant_message%'
               OR s.attributes LIKE '%session_id%'
               OR s.attributes LIKE '%agent_id%')
        ORDER BY t.start_time DESC
        """
        
        cursor.execute(query, (cutoff_date,))
        traces = cursor.fetchall()
        
        conversations = []
        
        for trace in traces:
            conversation = self._extract_conversation_from_trace(trace['trace_id'])
            if conversation and conversation.total_messages > 0:
                conversations.append(conversation)
        
        conn.close()
        return conversations
    
    def _extract_conversation_from_trace(self, trace_id: str) -> Optional[ConversationSession]:
        """Extract a complete conversation from a trace."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get trace info
        cursor.execute("SELECT * FROM traces WHERE trace_id = ?", (trace_id,))
        trace = cursor.fetchone()
        
        if not trace:
            conn.close()
            return None
        
        # Get all spans for this trace
        cursor.execute("""
            SELECT span_id, parent_span_id, name, start_time, end_time, attributes, status
            FROM spans 
            WHERE trace_id = ?
            ORDER BY start_time
        """, (trace_id,))
        
        spans = cursor.fetchall()
        conn.close()
        
        # Extract conversation data
        session_id = None
        agent_id = None
        user_messages = []
        assistant_messages = []
        tool_calls = 0
        errors = []
        topics = set()
        
        for span in spans:
            if not span['attributes']:
                continue
            
            try:
                attrs = json.loads(span['attributes'])
                
                # Extract session and agent IDs
                if 'session_id' in attrs:
                    session_id = attrs['session_id']
                if 'agent_id' in attrs:
                    agent_id = attrs['agent_id']
                
                # Extract messages
                if 'user_message' in attrs:
                    message = str(attrs['user_message'])
                    user_messages.append(message)
                    topics.update(self._extract_topics(message))
                
                if 'assistant_message' in attrs:
                    message = str(attrs['assistant_message'])
                    assistant_messages.append(message)
                    topics.update(self._extract_topics(message))
                
                # Count tool calls
                if 'tool_name' in attrs:
                    tool_calls += 1
                
                # Track errors
                if span['status'] == 'error' or 'error' in attrs:
                    error_msg = attrs.get('error_message', f"Error in {span['name']}")
                    errors.append(error_msg)
                    
            except json.JSONDecodeError:
                continue
        
        # Calculate duration
        duration_ms = 0.0
        if trace['start_time'] and trace['end_time']:
            try:
                start = datetime.fromisoformat(trace['start_time'].replace('Z', '+00:00'))
                end = datetime.fromisoformat(trace['end_time'].replace('Z', '+00:00'))
                duration_ms = (end - start).total_seconds() * 1000
            except:
                pass
        
        return ConversationSession(
            session_id=session_id,
            agent_id=agent_id,
            trace_ids=[trace_id],
            start_time=trace['start_time'],
            end_time=trace['end_time'],
            total_turns=min(len(user_messages), len(assistant_messages)),
            total_messages=len(user_messages) + len(assistant_messages),
            total_tool_calls=tool_calls,
            duration_ms=duration_ms,
            user_messages=user_messages,
            assistant_messages=assistant_messages,
            topics=topics,
            errors=errors
        )
    
    def _extract_topics(self, message: str) -> Set[str]:
        """Extract potential topics from a message (simple keyword extraction)."""
        # Simple topic extraction - you could enhance this with NLP
        topics = set()
        
        # Common topic keywords
        topic_keywords = {
            'code': ['code', 'programming', 'function', 'variable', 'debug'],
            'math': ['calculate', 'equation', 'number', 'math', 'formula'],
            'help': ['help', 'how to', 'explain', 'what is', 'tutorial'],
            'data': ['data', 'database', 'query', 'table', 'analysis'],
            'web': ['website', 'html', 'css', 'javascript', 'web'],
            'ai': ['ai', 'machine learning', 'model', 'training', 'neural'],
        }
        
        message_lower = message.lower()
        for topic, keywords in topic_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                topics.add(topic)
        
        return topics
    
    def analyze_conversation_patterns(self, conversations: List[ConversationSession]) -> ConversationAnalytics:
        """Analyze patterns across all conversations."""
        if not conversations:
            return ConversationAnalytics(
                total_conversations=0,
                total_messages=0,
                avg_conversation_length=0.0,
                avg_response_time_ms=0.0,
                most_common_topics=[],
                peak_usage_hours=[],
                error_rate=0.0,
                user_engagement_patterns={}
            )
        
        # Basic statistics
        total_conversations = len(conversations)
        total_messages = sum(conv.total_messages for conv in conversations)
        avg_length = total_messages / total_conversations if total_conversations > 0 else 0
        
        # Response time analysis
        valid_durations = [conv.duration_ms for conv in conversations if conv.duration_ms > 0]
        avg_response_time = sum(valid_durations) / len(valid_durations) if valid_durations else 0
        
        # Topic analysis
        topic_counts = {}
        for conv in conversations:
            for topic in conv.topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        most_common_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Peak usage analysis
        hour_counts = {}
        for conv in conversations:
            try:
                dt = datetime.fromisoformat(conv.start_time.replace('Z', '+00:00'))
                hour = dt.hour
                hour_counts[hour] = hour_counts.get(hour, 0) + 1
            except:
                continue
        
        peak_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        peak_usage_hours = [hour for hour, count in peak_hours]
        
        # Error rate
        conversations_with_errors = sum(1 for conv in conversations if conv.errors)
        error_rate = conversations_with_errors / total_conversations if total_conversations > 0 else 0
        
        # User engagement patterns
        turn_counts = [conv.total_turns for conv in conversations if conv.total_turns > 0]
        engagement_patterns = {
            'avg_turns_per_conversation': sum(turn_counts) / len(turn_counts) if turn_counts else 0,
            'max_turns': max(turn_counts) if turn_counts else 0,
            'conversations_with_tools': sum(1 for conv in conversations if conv.total_tool_calls > 0),
            'avg_tools_per_conversation': sum(conv.total_tool_calls for conv in conversations) / total_conversations if total_conversations > 0 else 0
        }
        
        return ConversationAnalytics(
            total_conversations=total_conversations,
            total_messages=total_messages,
            avg_conversation_length=avg_length,
            avg_response_time_ms=avg_response_time,
            most_common_topics=most_common_topics,
            peak_usage_hours=peak_usage_hours,
            error_rate=error_rate,
            user_engagement_patterns=engagement_patterns
        )
    
    def export_conversations(self, conversations: List[ConversationSession], format: str = 'json', output_file: str = None):
        """Export conversations to various formats."""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"conversations_{timestamp}.{format}"
        
        if format == 'json':
            self._export_to_json(conversations, output_file)
        elif format == 'csv':
            self._export_to_csv(conversations, output_file)
        elif format == 'markdown':
            self._export_to_markdown(conversations, output_file)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"✅ Exported {len(conversations)} conversations to {output_file}")
    
    def _export_to_json(self, conversations: List[ConversationSession], output_file: str):
        """Export to JSON format."""
        data = []
        for conv in conversations:
            conv_dict = asdict(conv)
            conv_dict['topics'] = list(conv_dict['topics'])  # Convert set to list
            data.append(conv_dict)
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _export_to_csv(self, conversations: List[ConversationSession], output_file: str):
        """Export to CSV format."""
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'session_id', 'agent_id', 'start_time', 'end_time', 'duration_ms',
                'total_turns', 'total_messages', 'total_tool_calls', 'topics', 'has_errors'
            ])
            
            # Data
            for conv in conversations:
                writer.writerow([
                    conv.session_id or '',
                    conv.agent_id or '',
                    conv.start_time,
                    conv.end_time,
                    conv.duration_ms,
                    conv.total_turns,
                    conv.total_messages,
                    conv.total_tool_calls,
                    ', '.join(conv.topics),
                    len(conv.errors) > 0
                ])
    
    def _export_to_markdown(self, conversations: List[ConversationSession], output_file: str):
        """Export to Markdown format."""
        with open(output_file, 'w') as f:
            f.write("# Conversation History Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for i, conv in enumerate(conversations, 1):
                f.write(f"## Conversation {i}\n\n")
                f.write(f"- **Session ID**: {conv.session_id or 'N/A'}\n")
                f.write(f"- **Agent ID**: {conv.agent_id or 'N/A'}\n")
                f.write(f"- **Time**: {conv.start_time} to {conv.end_time}\n")
                f.write(f"- **Duration**: {conv.duration_ms:.0f}ms\n")
                f.write(f"- **Turns**: {conv.total_turns}\n")
                f.write(f"- **Messages**: {conv.total_messages}\n")
                f.write(f"- **Tool Calls**: {conv.total_tool_calls}\n")
                f.write(f"- **Topics**: {', '.join(conv.topics) if conv.topics else 'None'}\n")
                
                if conv.errors:
                    f.write(f"- **Errors**: {len(conv.errors)}\n")
                
                f.write("\n### Messages\n\n")
                
                # Interleave user and assistant messages
                max_messages = max(len(conv.user_messages), len(conv.assistant_messages))
                for j in range(max_messages):
                    if j < len(conv.user_messages):
                        f.write(f"**User**: {conv.user_messages[j]}\n\n")
                    if j < len(conv.assistant_messages):
                        f.write(f"**Assistant**: {conv.assistant_messages[j]}\n\n")
                
                f.write("---\n\n")
    
    def print_analytics_report(self, analytics: ConversationAnalytics):
        """Print a comprehensive analytics report."""
        print("\n📊 Conversation Analytics Report")
        print("=" * 50)
        
        print(f"\n📈 Overview:")
        print(f"  • Total conversations: {analytics.total_conversations}")
        print(f"  • Total messages: {analytics.total_messages}")
        print(f"  • Average conversation length: {analytics.avg_conversation_length:.1f} messages")
        print(f"  • Average response time: {analytics.avg_response_time_ms:.0f}ms")
        print(f"  • Error rate: {analytics.error_rate:.1%}")
        
        if analytics.most_common_topics:
            print(f"\n🏷️  Most Common Topics:")
            for topic, count in analytics.most_common_topics:
                print(f"  • {topic}: {count} conversations")
        
        if analytics.peak_usage_hours:
            print(f"\n⏰ Peak Usage Hours:")
            for hour in analytics.peak_usage_hours:
                print(f"  • {hour:02d}:00")
        
        print(f"\n👥 User Engagement:")
        engagement = analytics.user_engagement_patterns
        print(f"  • Average turns per conversation: {engagement['avg_turns_per_conversation']:.1f}")
        print(f"  • Maximum turns in a conversation: {engagement['max_turns']}")
        print(f"  • Conversations with tool usage: {engagement['conversations_with_tools']}")
        print(f"  • Average tools per conversation: {engagement['avg_tools_per_conversation']:.1f}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Analyze Llama Stack conversation histories")
    parser.add_argument(
        "--db",
        default="/Users/azaalouk/.llama/distributions/distribution-myenv-ollama/trace_store.db",
        help="Path to the telemetry database"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze conversation patterns")
    analyze_parser.add_argument("--days", type=int, default=30, help="Number of days to analyze")
    analyze_parser.add_argument("--export", choices=['json', 'csv', 'markdown'], help="Export format")
    analyze_parser.add_argument("--output", help="Output file name")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export conversation data")
    export_parser.add_argument("--days", type=int, default=30, help="Number of days to export")
    export_parser.add_argument("--format", choices=['json', 'csv', 'markdown'], default='json', help="Export format")
    export_parser.add_argument("--output", help="Output file name")
    
    # Session command
    session_parser = subparsers.add_parser("session", help="Analyze a specific session")
    session_parser.add_argument("session_id", help="Session ID to analyze")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        analyzer = MessageHistoryAnalyzer(args.db)
        
        if args.command == "analyze":
            print(f"🔍 Analyzing conversations from the last {args.days} days...")
            conversations = analyzer.get_all_conversations(args.days)
            
            if not conversations:
                print("📭 No conversations found in the specified time period.")
                return
            
            analytics = analyzer.analyze_conversation_patterns(conversations)
            analyzer.print_analytics_report(analytics)
            
            if args.export:
                analyzer.export_conversations(conversations, args.export, args.output)
        
        elif args.command == "export":
            print(f"📤 Exporting conversations from the last {args.days} days...")
            conversations = analyzer.get_all_conversations(args.days)
            
            if not conversations:
                print("📭 No conversations found to export.")
                return
            
            analyzer.export_conversations(conversations, args.format, args.output)
        
        elif args.command == "session":
            print(f"🔍 Analyzing session: {args.session_id}")
            # This would need additional implementation to filter by session ID
            print("Session-specific analysis not yet implemented.")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
