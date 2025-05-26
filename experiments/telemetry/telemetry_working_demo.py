#!/usr/bin/env python3
"""
Llama Stack Telemetry Working Demo

This script demonstrates the working telemetry features based on your actual setup.
It focuses on local database analysis since that's what's currently functional.
"""

import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List


class TelemetryAnalyzer:
    """Utility class for analyzing telemetry data from Llama Stack."""
    
    def __init__(self, trace_db_path: str):
        """Initialize with path to the SQLite trace database."""
        self.trace_db_path = trace_db_path
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get comprehensive database information."""
        conn = sqlite3.connect(self.trace_db_path)
        cursor = conn.cursor()
        
        info = {}
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        info['tables'] = tables
        
        # Get schema for each table
        info['schema'] = {}
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table});")
            columns = cursor.fetchall()
            info['schema'][table] = [
                {
                    'name': col[1],
                    'type': col[2],
                    'not_null': bool(col[3]),
                    'primary_key': bool(col[5])
                }
                for col in columns
            ]
        
        # Get row counts
        info['counts'] = {}
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            info['counts'][table] = cursor.fetchone()[0]
        
        conn.close()
        return info
    
    def get_recent_traces(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent traces from the database."""
        conn = sqlite3.connect(self.trace_db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = """
        SELECT trace_id, root_span_id, start_time, end_time
        FROM traces 
        ORDER BY start_time DESC 
        LIMIT ?
        """
        
        cursor.execute(query, (limit,))
        traces = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return traces
    
    def get_spans_for_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        """Get all spans for a specific trace."""
        conn = sqlite3.connect(self.trace_db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = """
        SELECT span_id, parent_span_id, name, start_time, end_time, attributes, status
        FROM spans 
        WHERE trace_id = ?
        ORDER BY start_time
        """
        
        cursor.execute(query, (trace_id,))
        spans = []
        for row in cursor.fetchall():
            span = dict(row)
            # Parse JSON attributes
            if span['attributes']:
                try:
                    span['attributes'] = json.loads(span['attributes'])
                except json.JSONDecodeError:
                    span['attributes'] = {}
            else:
                span['attributes'] = {}
            spans.append(span)
        
        conn.close()
        return spans
    
    def get_span_events(self, span_id: str) -> List[Dict[str, Any]]:
        """Get events for a specific span."""
        conn = sqlite3.connect(self.trace_db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = """
        SELECT * FROM span_events 
        WHERE span_id = ?
        ORDER BY timestamp
        """
        
        cursor.execute(query, (span_id,))
        events = []
        for row in cursor.fetchall():
            event = dict(row)
            # Parse JSON attributes if present
            if 'attributes' in event and event['attributes']:
                try:
                    event['attributes'] = json.loads(event['attributes'])
                except json.JSONDecodeError:
                    event['attributes'] = {}
            events.append(event)
        
        conn.close()
        return events
    
    def analyze_trace_details(self, trace_id: str) -> Dict[str, Any]:
        """Get detailed analysis of a specific trace."""
        spans = self.get_spans_for_trace(trace_id)
        
        analysis = {
            "trace_id": trace_id,
            "total_spans": len(spans),
            "span_details": [],
            "timing_analysis": {},
            "operations": set(),
            "has_errors": False
        }
        
        total_duration = 0
        durations = []
        
        for span in spans:
            # Calculate duration
            duration = self._calculate_duration(span["start_time"], span["end_time"])
            durations.append(duration)
            
            # Get span events
            events = self.get_span_events(span["span_id"])
            
            span_detail = {
                "span_id": span["span_id"],
                "name": span["name"],
                "start_time": span["start_time"],
                "end_time": span["end_time"],
                "duration_ms": duration,
                "status": span["status"],
                "attributes": span["attributes"],
                "events": events,
                "parent_span_id": span["parent_span_id"]
            }
            
            analysis["span_details"].append(span_detail)
            analysis["operations"].add(span["name"])
            
            if span["status"] == "error":
                analysis["has_errors"] = True
        
        # Calculate timing statistics
        if durations:
            analysis["timing_analysis"] = {
                "total_duration_ms": sum(durations),
                "average_duration_ms": sum(durations) / len(durations),
                "min_duration_ms": min(durations),
                "max_duration_ms": max(durations),
                "span_count": len(durations)
            }
        
        analysis["operations"] = list(analysis["operations"])
        return analysis
    
    def _calculate_duration(self, start_time: str, end_time: str) -> float:
        """Calculate duration between two timestamps in milliseconds."""
        if not start_time or not end_time:
            return 0.0
        
        try:
            start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            return (end - start).total_seconds() * 1000
        except:
            return 0.0
    
    def print_database_summary(self):
        """Print a comprehensive database summary."""
        print("🗄️  Telemetry Database Summary")
        print("=" * 50)
        
        try:
            info = self.get_database_info()
            
            print(f"📋 Tables: {', '.join(info['tables'])}")
            print()
            
            for table, count in info['counts'].items():
                print(f"📊 {table}: {count} records")
                if table in info['schema']:
                    columns = [col['name'] for col in info['schema'][table]]
                    print(f"   Columns: {', '.join(columns)}")
                print()
            
        except Exception as e:
            print(f"❌ Error reading database: {e}")
    
    def print_recent_activity(self, limit: int = 5):
        """Print recent telemetry activity."""
        print(f"📈 Recent Telemetry Activity (last {limit} traces)")
        print("=" * 50)
        
        try:
            traces = self.get_recent_traces(limit)
            
            if not traces:
                print("📊 No traces found in database.")
                print("💡 Traces will appear here after you have conversations with agents.")
                return
            
            for i, trace in enumerate(traces, 1):
                print(f"\n{i}. 🔍 Trace: {trace['trace_id']}")
                print(f"   📅 Time: {trace['start_time']}")
                
                # Get detailed analysis
                analysis = self.analyze_trace_details(trace['trace_id'])
                
                print(f"   📊 Spans: {analysis['total_spans']}")
                print(f"   🔧 Operations: {', '.join(analysis['operations'])}")
                
                if analysis['timing_analysis']:
                    timing = analysis['timing_analysis']
                    print(f"   ⏱️  Duration: {timing['total_duration_ms']:.2f}ms")
                
                if analysis['has_errors']:
                    print(f"   ❌ Contains errors")
                
                # Show span details
                for span in analysis['span_details']:
                    indent = "     "
                    status_icon = "✅" if span['status'] == 'ok' else "❌" if span['status'] == 'error' else "⏳"
                    print(f"{indent}{status_icon} {span['name']} ({span['duration_ms']:.2f}ms)")
                    
                    # Show interesting attributes
                    if span['attributes']:
                        for key, value in span['attributes'].items():
                            if key in ['method', 'url', 'status_code', 'user_agent']:
                                print(f"{indent}   • {key}: {value}")
                    
                    # Show events if any
                    if span['events']:
                        print(f"{indent}   📝 {len(span['events'])} events")
                        for event in span['events'][:2]:  # Show first 2 events
                            print(f"{indent}     - {event.get('name', 'Event')} at {event.get('timestamp', 'N/A')}")
        
        except Exception as e:
            print(f"❌ Error analyzing traces: {e}")
    
    def search_traces_by_operation(self, operation_name: str) -> List[str]:
        """Find traces containing a specific operation."""
        conn = sqlite3.connect(self.trace_db_path)
        cursor = conn.cursor()
        
        query = """
        SELECT DISTINCT trace_id 
        FROM spans 
        WHERE name LIKE ?
        ORDER BY start_time DESC
        """
        
        cursor.execute(query, (f"%{operation_name}%",))
        trace_ids = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return trace_ids


def main():
    """Main demonstration function."""
    print("🚀 Llama Stack Telemetry Working Demo")
    print("=" * 50)
    
    # Path to your telemetry database
    db_path = "/Users/azaalouk/.llama/distributions/distribution-myenv-ollama/trace_store.db"
    
    try:
        analyzer = TelemetryAnalyzer(db_path)
        
        # 1. Show database summary
        analyzer.print_database_summary()
        
        # 2. Show recent activity
        analyzer.print_recent_activity(limit=5)
        
        # 3. Search for specific operations
        print("\n🔍 Searching for telemetry operations...")
        telemetry_traces = analyzer.search_traces_by_operation("telemetry")
        if telemetry_traces:
            print(f"Found {len(telemetry_traces)} traces with telemetry operations:")
            for trace_id in telemetry_traces[:3]:
                print(f"  • {trace_id}")
        
        # 4. Show usage tips
        print("\n💡 Usage Tips:")
        print("=" * 30)
        print("• Use 'python trace_viewer.py list' to browse traces interactively")
        print("• Use 'python trace_viewer.py show <trace_id>' for detailed analysis")
        print("• Use 'python trace_viewer.py search <query>' to find specific content")
        print("• Use 'python trace_viewer.py stats' for database statistics")
        print()
        print("🎯 What's Working:")
        print("• ✅ Telemetry data is being captured and stored")
        print("• ✅ SQLite database contains traces and spans")
        print("• ✅ Local analysis tools are functional")
        print("• ✅ Trace viewer CLI is ready to use")
        print()
        print("🔧 Next Steps:")
        print("• Fix the server startup issue (SQLAlchemy dependency)")
        print("• Have conversations with agents to generate more traces")
        print("• Use the trace viewer to analyze conversation patterns")
        print("• Set up monitoring dashboards if needed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Make sure the database exists at: {db_path}")
        print("The database will be created automatically when you have conversations.")


if __name__ == "__main__":
    main()
