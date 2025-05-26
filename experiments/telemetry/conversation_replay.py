#!/usr/bin/env python3
"""
Conversation Replay Tool for Llama Stack Telemetry

This tool allows you to replay entire conversations step-by-step,
showing the full message history and conversation flow.
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ConversationMessage:
    """Represents a single message in a conversation."""
    timestamp: str
    role: str  # 'user', 'assistant', 'system', 'tool'
    content: str
    metadata: Dict[str, Any]
    span_id: str
    trace_id: str


@dataclass
class ConversationTurn:
    """Represents a complete conversation turn (user message + assistant response)."""
    turn_number: int
    user_message: Optional[ConversationMessage]
    assistant_message: Optional[ConversationMessage]
    tool_calls: List[ConversationMessage]
    duration_ms: float
    timestamp: str


class ConversationExtractor:
    """Extracts and reconstructs conversations from telemetry data."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def get_conversation_traces(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get traces that contain conversation data."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Look for traces with agent or conversation-related spans
        query = """
        SELECT DISTINCT t.trace_id, t.root_span_id, t.start_time, t.end_time,
               COUNT(s.span_id) as span_count
        FROM traces t
        JOIN spans s ON t.trace_id = s.trace_id
        WHERE s.name LIKE '%agent%' OR s.name LIKE '%turn%' OR s.name LIKE '%conversation%'
           OR s.attributes LIKE '%user_message%' OR s.attributes LIKE '%assistant_message%'
        GROUP BY t.trace_id, t.root_span_id, t.start_time, t.end_time
        ORDER BY t.start_time DESC
        LIMIT ?
        """
        
        cursor.execute(query, (limit,))
        traces = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return traces
    
    def extract_messages_from_trace(self, trace_id: str) -> List[ConversationMessage]:
        """Extract all conversation messages from a trace."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get all spans for this trace
        query = """
        SELECT span_id, parent_span_id, name, start_time, end_time, attributes, status
        FROM spans 
        WHERE trace_id = ?
        ORDER BY start_time
        """
        
        cursor.execute(query, (trace_id,))
        spans = cursor.fetchall()
        
        messages = []
        
        for span in spans:
            if not span['attributes']:
                continue
            
            try:
                attrs = json.loads(span['attributes'])
                
                # Extract user messages
                if 'user_message' in attrs:
                    message = ConversationMessage(
                        timestamp=span['start_time'],
                        role='user',
                        content=str(attrs['user_message']),
                        metadata=attrs,
                        span_id=span['span_id'],
                        trace_id=trace_id
                    )
                    messages.append(message)
                
                # Extract assistant messages
                if 'assistant_message' in attrs:
                    message = ConversationMessage(
                        timestamp=span['end_time'] or span['start_time'],
                        role='assistant',
                        content=str(attrs['assistant_message']),
                        metadata=attrs,
                        span_id=span['span_id'],
                        trace_id=trace_id
                    )
                    messages.append(message)
                
                # Extract tool calls
                if 'tool_name' in attrs:
                    tool_content = f"Tool: {attrs['tool_name']}"
                    if 'tool_arguments' in attrs:
                        tool_content += f"\nArguments: {attrs['tool_arguments']}"
                    if 'tool_result' in attrs:
                        tool_content += f"\nResult: {attrs['tool_result']}"
                    
                    message = ConversationMessage(
                        timestamp=span['start_time'],
                        role='tool',
                        content=tool_content,
                        metadata=attrs,
                        span_id=span['span_id'],
                        trace_id=trace_id
                    )
                    messages.append(message)
                
                # Extract system messages or other content
                if 'content' in attrs and 'user_message' not in attrs and 'assistant_message' not in attrs:
                    message = ConversationMessage(
                        timestamp=span['start_time'],
                        role='system',
                        content=str(attrs['content']),
                        metadata=attrs,
                        span_id=span['span_id'],
                        trace_id=trace_id
                    )
                    messages.append(message)
                    
            except json.JSONDecodeError:
                continue
        
        conn.close()
        
        # Sort messages by timestamp
        messages.sort(key=lambda m: m.timestamp)
        return messages
    
    def organize_into_turns(self, messages: List[ConversationMessage]) -> List[ConversationTurn]:
        """Organize messages into conversation turns."""
        turns = []
        current_turn = None
        turn_number = 1
        
        for message in messages:
            if message.role == 'user':
                # Start a new turn
                if current_turn:
                    turns.append(current_turn)
                
                current_turn = ConversationTurn(
                    turn_number=turn_number,
                    user_message=message,
                    assistant_message=None,
                    tool_calls=[],
                    duration_ms=0.0,
                    timestamp=message.timestamp
                )
                turn_number += 1
            
            elif message.role == 'assistant' and current_turn:
                current_turn.assistant_message = message
                # Calculate turn duration
                if current_turn.user_message:
                    current_turn.duration_ms = self._calculate_duration(
                        current_turn.user_message.timestamp,
                        message.timestamp
                    )
            
            elif message.role == 'tool' and current_turn:
                current_turn.tool_calls.append(message)
        
        # Add the last turn
        if current_turn:
            turns.append(current_turn)
        
        return turns
    
    def _calculate_duration(self, start_time: str, end_time: str) -> float:
        """Calculate duration between two timestamps in milliseconds."""
        try:
            start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            return (end - start).total_seconds() * 1000
        except:
            return 0.0


class ConversationReplayer:
    """Interactive conversation replay interface."""
    
    def __init__(self, db_path: str):
        self.extractor = ConversationExtractor(db_path)
        self.colors = {
            'user': '\033[94m',      # Blue
            'assistant': '\033[92m',  # Green
            'tool': '\033[93m',       # Yellow
            'system': '\033[95m',     # Magenta
            'reset': '\033[0m',       # Reset
            'bold': '\033[1m',        # Bold
            'dim': '\033[2m'          # Dim
        }
    
    def list_conversations(self, limit: int = 20):
        """List available conversations."""
        print(f"\n🎬 Available Conversations (last {limit})")
        print("=" * 60)
        
        traces = self.extractor.get_conversation_traces(limit)
        
        if not traces:
            print("📭 No conversation traces found.")
            print("💡 Conversations will appear here after you have agent interactions.")
            return []
        
        for i, trace in enumerate(traces, 1):
            # Try to get a preview of the conversation
            messages = self.extractor.extract_messages_from_trace(trace['trace_id'])
            
            print(f"{i:2d}. 🔍 {trace['trace_id'][:16]}...")
            print(f"    📅 {trace['start_time']}")
            print(f"    📊 {trace['span_count']} spans, {len(messages)} messages")
            
            # Show first user message as preview
            user_messages = [m for m in messages if m.role == 'user']
            if user_messages:
                preview = user_messages[0].content[:80]
                print(f"    💬 \"{preview}{'...' if len(user_messages[0].content) > 80 else ''}\"")
            print()
        
        return traces
    
    def replay_conversation(self, trace_id: str, interactive: bool = True):
        """Replay a conversation step by step."""
        print(f"\n🎬 Replaying Conversation: {trace_id}")
        print("=" * 60)
        
        messages = self.extractor.extract_messages_from_trace(trace_id)
        
        if not messages:
            print("📭 No conversation messages found in this trace.")
            return
        
        turns = self.extractor.organize_into_turns(messages)
        
        if not turns:
            print("📭 Could not organize messages into conversation turns.")
            print("📊 Raw messages found:")
            for msg in messages:
                self._print_message(msg)
            return
        
        print(f"📊 Found {len(turns)} conversation turns")
        print(f"⏱️  Total conversation time: {self._format_timestamp(messages[0].timestamp)} to {self._format_timestamp(messages[-1].timestamp)}")
        
        if interactive:
            print(f"\n{self.colors['dim']}Press Enter to see each turn, 'q' to quit, 's' to skip to end{self.colors['reset']}")
            input("Press Enter to start...")
        
        for i, turn in enumerate(turns):
            self._replay_turn(turn, interactive)
            
            if interactive:
                user_input = input(f"\n{self.colors['dim']}[Turn {i+1}/{len(turns)}] Press Enter for next turn, 'q' to quit: {self.colors['reset']}")
                if user_input.lower() == 'q':
                    break
                elif user_input.lower() == 's':
                    interactive = False
        
        print(f"\n🎉 Conversation replay complete!")
        self._print_conversation_summary(turns)
    
    def _replay_turn(self, turn: ConversationTurn, show_metadata: bool = False):
        """Replay a single conversation turn."""
        print(f"\n{self.colors['bold']}--- Turn {turn.turn_number} ({self._format_timestamp(turn.timestamp)}) ---{self.colors['reset']}")
        
        # Show user message
        if turn.user_message:
            self._print_message(turn.user_message, show_metadata)
        
        # Show tool calls
        for tool_call in turn.tool_calls:
            self._print_message(tool_call, show_metadata)
        
        # Show assistant response
        if turn.assistant_message:
            self._print_message(turn.assistant_message, show_metadata)
            
            if turn.duration_ms > 0:
                print(f"{self.colors['dim']}⏱️  Response time: {turn.duration_ms:.0f}ms{self.colors['reset']}")
    
    def _print_message(self, message: ConversationMessage, show_metadata: bool = False):
        """Print a single message with formatting."""
        color = self.colors.get(message.role, self.colors['reset'])
        role_icon = {
            'user': '👤',
            'assistant': '🤖',
            'tool': '🔧',
            'system': '⚙️'
        }.get(message.role, '💬')
        
        print(f"\n{color}{self.colors['bold']}{role_icon} {message.role.title()}:{self.colors['reset']}")
        
        # Format content with proper wrapping
        content_lines = message.content.split('\n')
        for line in content_lines:
            if len(line) > 80:
                # Wrap long lines
                words = line.split(' ')
                current_line = ""
                for word in words:
                    if len(current_line + word) > 80:
                        print(f"{color}{current_line}{self.colors['reset']}")
                        current_line = word + " "
                    else:
                        current_line += word + " "
                if current_line:
                    print(f"{color}{current_line.rstrip()}{self.colors['reset']}")
            else:
                print(f"{color}{line}{self.colors['reset']}")
        
        if show_metadata and message.metadata:
            print(f"{self.colors['dim']}📋 Metadata: {json.dumps(message.metadata, indent=2)}{self.colors['reset']}")
    
    def _format_timestamp(self, timestamp: str) -> str:
        """Format timestamp for display."""
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime("%H:%M:%S")
        except:
            return timestamp
    
    def _print_conversation_summary(self, turns: List[ConversationTurn]):
        """Print a summary of the conversation."""
        print(f"\n📊 Conversation Summary")
        print("=" * 30)
        
        total_user_messages = sum(1 for turn in turns if turn.user_message)
        total_assistant_messages = sum(1 for turn in turns if turn.assistant_message)
        total_tool_calls = sum(len(turn.tool_calls) for turn in turns)
        avg_response_time = sum(turn.duration_ms for turn in turns if turn.duration_ms > 0) / len([t for t in turns if t.duration_ms > 0]) if turns else 0
        
        print(f"🔢 Total turns: {len(turns)}")
        print(f"👤 User messages: {total_user_messages}")
        print(f"🤖 Assistant messages: {total_assistant_messages}")
        print(f"🔧 Tool calls: {total_tool_calls}")
        if avg_response_time > 0:
            print(f"⏱️  Average response time: {avg_response_time:.0f}ms")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Replay Llama Stack conversations")
    parser.add_argument(
        "--db",
        default="/Users/azaalouk/.llama/distributions/distribution-myenv-ollama/trace_store.db",
        help="Path to the telemetry database"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available conversations")
    list_parser.add_argument("--limit", type=int, default=20, help="Number of conversations to show")
    
    # Replay command
    replay_parser = subparsers.add_parser("replay", help="Replay a specific conversation")
    replay_parser.add_argument("trace_id", help="Trace ID to replay")
    replay_parser.add_argument("--no-interactive", action="store_true", help="Show all at once without pausing")
    replay_parser.add_argument("--metadata", action="store_true", help="Show message metadata")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        replayer = ConversationReplayer(args.db)
        
        if args.command == "list":
            traces = replayer.list_conversations(args.limit)
            if traces:
                print(f"\n💡 Use 'python conversation_replay.py replay <trace_id>' to replay a conversation")
        
        elif args.command == "replay":
            replayer.replay_conversation(
                args.trace_id,
                interactive=not args.no_interactive
            )
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
