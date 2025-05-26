#!/usr/bin/env python3
"""
Simple Trace Viewer for Llama Stack Telemetry

A command-line tool to browse and analyze your conversation traces.
"""

import argparse
import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional


class TraceViewer:
    """Interactive trace viewer for Llama Stack telemetry data."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._check_database()
    
    def _check_database(self):
        """Check if the database exists and has the expected schema."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check for required tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            if 'traces' not in tables or 'spans' not in tables:
                raise ValueError(f"Database at {self.db_path} doesn't have required telemetry tables")
            
            conn.close()
            print(f"✅ Connected to telemetry database: {self.db_path}")
            
        except sqlite3.Error as e:
            raise ValueError(f"Cannot access database at {self.db_path}: {e}")
    
    def list_traces(self, limit: int = 20, show_details: bool = False) -> List[Dict[str, Any]]:
        """List recent traces."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = """
        SELECT t.trace_id, t.root_span_id, t.start_time, t.end_time,
               COUNT(s.span_id) as span_count
        FROM traces t
        LEFT JOIN spans s ON t.trace_id = s.trace_id
        GROUP BY t.trace_id, t.root_span_id, t.start_time, t.end_time
        ORDER BY t.start_time DESC
        LIMIT ?
        """
        
        cursor.execute(query, (limit,))
        traces = []
        
        print(f"\n📊 Recent Traces (showing {limit} most recent):")
        print("=" * 80)
        
        for i, row in enumerate(cursor.fetchall(), 1):
            trace = dict(row)
            traces.append(trace)
            
            # Calculate duration
            duration = "N/A"
            if trace['start_time'] and trace['end_time']:
                start = datetime.fromisoformat(trace['start_time'])
                end = datetime.fromisoformat(trace['end_time'])
                duration_ms = (end - start).total_seconds() * 1000
                duration = f"{duration_ms:.0f}ms"
            
            print(f"{i:2d}. 🔍 {trace['trace_id'][:16]}... ({trace['span_count']} spans, {duration})")
            print(f"    📅 {trace['start_time']}")
            
            if show_details:
                # Get first span to show what this trace is about
                cursor.execute("""
                    SELECT name, attributes FROM spans 
                    WHERE trace_id = ? AND parent_span_id IS NULL
                    LIMIT 1
                """, (trace['trace_id'],))
                
                root_span = cursor.fetchone()
                if root_span:
                    print(f"    📝 {root_span['name']}")
                    if root_span['attributes']:
                        try:
                            attrs = json.loads(root_span['attributes'])
                            if 'user_message' in attrs:
                                msg = str(attrs['user_message'])[:60]
                                print(f"    💬 \"{msg}{'...' if len(str(attrs['user_message'])) > 60 else ''}\"")
                        except:
                            pass
            print()
        
        conn.close()
        return traces
    
    def show_trace_details(self, trace_id: str):
        """Show detailed information about a specific trace."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get trace info
        cursor.execute("SELECT * FROM traces WHERE trace_id = ?", (trace_id,))
        trace = cursor.fetchone()
        
        if not trace:
            print(f"❌ Trace {trace_id} not found")
            return
        
        print(f"\n🔍 Trace Details: {trace_id}")
        print("=" * 80)
        print(f"📅 Start: {trace['start_time']}")
        print(f"📅 End: {trace['end_time']}")
        
        # Calculate total duration
        if trace['start_time'] and trace['end_time']:
            start = datetime.fromisoformat(trace['start_time'])
            end = datetime.fromisoformat(trace['end_time'])
            duration_ms = (end - start).total_seconds() * 1000
            print(f"⏱️  Duration: {duration_ms:.2f}ms")
        
        # Get all spans for this trace
        cursor.execute("""
            SELECT span_id, parent_span_id, name, start_time, end_time, attributes, status
            FROM spans 
            WHERE trace_id = ?
            ORDER BY start_time
        """, (trace_id,))
        
        spans = cursor.fetchall()
        print(f"\n📊 Spans ({len(spans)} total):")
        
        # Build span hierarchy
        span_dict = {}
        root_spans = []
        
        for span in spans:
            span_data = dict(span)
            span_data['children'] = []
            span_dict[span['span_id']] = span_data
            
            if span['parent_span_id'] is None:
                root_spans.append(span_data)
            else:
                parent = span_dict.get(span['parent_span_id'])
                if parent:
                    parent['children'].append(span_data)
        
        # Print hierarchy
        for root_span in root_spans:
            self._print_span_tree(root_span, 0)
        
        conn.close()
    
    def _print_span_tree(self, span: Dict[str, Any], depth: int):
        """Print a span and its children in a tree format."""
        indent = "  " * depth
        
        # Calculate duration
        duration = "N/A"
        if span['start_time'] and span['end_time']:
            start = datetime.fromisoformat(span['start_time'])
            end = datetime.fromisoformat(span['end_time'])
            duration_ms = (end - start).total_seconds() * 1000
            duration = f"{duration_ms:.2f}ms"
        
        # Status indicator
        status_icon = "✅" if span['status'] == 'ok' else "❌" if span['status'] == 'error' else "⏳"
        
        print(f"{indent}{status_icon} {span['name']} ({duration})")
        
        # Show relevant attributes
        if span['attributes']:
            try:
                attrs = json.loads(span['attributes'])
                for key, value in attrs.items():
                    if key in ['model', 'user_message', 'assistant_message', 'tool_name', 'error_message']:
                        value_str = str(value)
                        if len(value_str) > 80:
                            value_str = value_str[:77] + "..."
                        print(f"{indent}  • {key}: {value_str}")
            except:
                pass
        
        # Print children
        for child in span['children']:
            self._print_span_tree(child, depth + 1)
    
    def search_traces(self, query: str, limit: int = 10):
        """Search traces by content."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Search in span attributes
        search_query = """
        SELECT DISTINCT t.trace_id, t.start_time, s.name, s.attributes
        FROM traces t
        JOIN spans s ON t.trace_id = s.trace_id
        WHERE s.attributes LIKE ?
        ORDER BY t.start_time DESC
        LIMIT ?
        """
        
        cursor.execute(search_query, (f"%{query}%", limit))
        results = cursor.fetchall()
        
        print(f"\n🔍 Search Results for '{query}' (showing {len(results)} matches):")
        print("=" * 80)
        
        for i, row in enumerate(results, 1):
            print(f"{i:2d}. 🔍 {row['trace_id'][:16]}... - {row['name']}")
            print(f"    📅 {row['start_time']}")
            
            # Try to show matching content
            if row['attributes']:
                try:
                    attrs = json.loads(row['attributes'])
                    for key, value in attrs.items():
                        if query.lower() in str(value).lower():
                            value_str = str(value)
                            if len(value_str) > 100:
                                # Find the query in the string and show context
                                query_pos = value_str.lower().find(query.lower())
                                start = max(0, query_pos - 30)
                                end = min(len(value_str), query_pos + len(query) + 30)
                                value_str = "..." + value_str[start:end] + "..."
                            print(f"    💬 {key}: {value_str}")
                            break
                except:
                    pass
            print()
        
        conn.close()
    
    def show_stats(self):
        """Show database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        print("\n📊 Database Statistics:")
        print("=" * 40)
        
        # Total counts
        cursor.execute("SELECT COUNT(*) FROM traces")
        trace_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM spans")
        span_count = cursor.fetchone()[0]
        
        print(f"📈 Total traces: {trace_count}")
        print(f"📈 Total spans: {span_count}")
        
        if trace_count > 0:
            # Date range
            cursor.execute("SELECT MIN(start_time), MAX(start_time) FROM traces")
            min_date, max_date = cursor.fetchone()
            print(f"📅 Date range: {min_date} to {max_date}")
            
            # Average spans per trace
            avg_spans = span_count / trace_count
            print(f"📊 Average spans per trace: {avg_spans:.1f}")
            
            # Most common span names
            cursor.execute("""
                SELECT name, COUNT(*) as count 
                FROM spans 
                GROUP BY name 
                ORDER BY count DESC 
                LIMIT 5
            """)
            
            print(f"\n🏆 Most common operations:")
            for name, count in cursor.fetchall():
                print(f"  • {name}: {count}")
        
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Browse Llama Stack telemetry traces")
    parser.add_argument(
        "--db", 
        default="/Users/azaalouk/.llama/distributions/distribution-myenv-ollama/trace_store.db",
        help="Path to the SQLite telemetry database"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List recent traces")
    list_parser.add_argument("--limit", type=int, default=20, help="Number of traces to show")
    list_parser.add_argument("--details", action="store_true", help="Show additional details")
    
    # Show command
    show_parser = subparsers.add_parser("show", help="Show detailed trace information")
    show_parser.add_argument("trace_id", help="Trace ID to show")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search traces by content")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--limit", type=int, default=10, help="Number of results to show")
    
    # Stats command
    subparsers.add_parser("stats", help="Show database statistics")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        viewer = TraceViewer(args.db)
        
        if args.command == "list":
            traces = viewer.list_traces(args.limit, args.details)
            if traces:
                print(f"\n💡 Use 'python trace_viewer.py show <trace_id>' to see details")
                print(f"💡 Use 'python trace_viewer.py search <query>' to search traces")
        
        elif args.command == "show":
            viewer.show_trace_details(args.trace_id)
        
        elif args.command == "search":
            viewer.search_traces(args.query, args.limit)
        
        elif args.command == "stats":
            viewer.show_stats()
    
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
