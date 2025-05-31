#!/usr/bin/env python3
"""
OpenTelemetry Conversation Exporter for Llama Stack Telemetry

This tool exports conversation traces from Llama Stack's SQLite telemetry database
to OpenTelemetry format for import into other observability tools.
"""

import argparse
import json
import sqlite3
import sys
import gzip
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import time


@dataclass
class OTLPSpan:
    """Represents an OpenTelemetry span in OTLP format."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    kind: int
    start_time_unix_nano: int
    end_time_unix_nano: int
    attributes: List[Dict[str, Any]]
    status: Dict[str, Any]


@dataclass
class OTLPTrace:
    """Represents an OpenTelemetry trace in OTLP format."""
    resource: Dict[str, Any]
    scope_spans: List[Dict[str, Any]]


class ConversationOTLPExporter:
    """Exports Llama Stack conversations to OpenTelemetry format."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.service_name = "llama-stack-agent"
        self.service_version = "1.0.0"
    
    def get_conversation_traces(self, days: Optional[int] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get conversation traces from the database."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Build query with optional time filtering
        query = """
        SELECT DISTINCT t.trace_id, t.root_span_id, t.start_time, t.end_time,
               COUNT(s.span_id) as span_count
        FROM traces t
        JOIN spans s ON t.trace_id = s.trace_id
        WHERE s.name LIKE '%agent%' OR s.name LIKE '%turn%' OR s.name LIKE '%conversation%'
           OR s.attributes LIKE '%user_message%' OR s.attributes LIKE '%assistant_message%'
           OR s.attributes LIKE '%tool_name%'
        """
        
        params = []
        if days:
            cutoff_date = datetime.now() - timedelta(days=days)
            query += " AND t.start_time >= ?"
            params.append(cutoff_date.isoformat())
        
        query += """
        GROUP BY t.trace_id, t.root_span_id, t.start_time, t.end_time
        ORDER BY t.start_time DESC
        """
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        cursor.execute(query, params)
        traces = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return traces
    
    def get_spans_for_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        """Get all spans for a specific trace."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = """
        SELECT span_id, parent_span_id, name, start_time, end_time, attributes, status
        FROM spans 
        WHERE trace_id = ?
        ORDER BY start_time
        """
        
        cursor.execute(query, (trace_id,))
        spans = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return spans
    
    def convert_timestamp_to_nano(self, timestamp: str) -> int:
        """Convert ISO timestamp to nanoseconds since Unix epoch."""
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return int(dt.timestamp() * 1_000_000_000)
        except:
            # Fallback to current time if parsing fails
            return int(time.time() * 1_000_000_000)
    
    def generate_span_id(self, original_id: str) -> str:
        """Generate a valid 16-character hex span ID."""
        # Use hash of original ID to create consistent span IDs
        import hashlib
        hash_obj = hashlib.md5(original_id.encode())
        return hash_obj.hexdigest()[:16]
    
    def generate_trace_id(self, original_id: str) -> str:
        """Generate a valid 32-character hex trace ID."""
        # Use hash of original ID to create consistent trace IDs
        import hashlib
        hash_obj = hashlib.md5(original_id.encode())
        return hash_obj.hexdigest()
    
    def convert_span_to_otlp(self, span: Dict[str, Any], trace_id: str) -> OTLPSpan:
        """Convert a Llama Stack span to OTLP format."""
        # Parse attributes
        attributes = []
        if span['attributes']:
            try:
                attrs = json.loads(span['attributes'])
                for key, value in attrs.items():
                    # Convert to OTLP attribute format
                    if isinstance(value, str):
                        attributes.append({
                            "key": key,
                            "value": {"stringValue": value}
                        })
                    elif isinstance(value, (int, float)):
                        attributes.append({
                            "key": key,
                            "value": {"doubleValue": float(value)}
                        })
                    elif isinstance(value, bool):
                        attributes.append({
                            "key": key,
                            "value": {"boolValue": value}
                        })
                    else:
                        # Convert complex objects to JSON strings
                        attributes.append({
                            "key": key,
                            "value": {"stringValue": json.dumps(value)}
                        })
            except json.JSONDecodeError:
                # If attributes can't be parsed, add as raw string
                attributes.append({
                    "key": "raw_attributes",
                    "value": {"stringValue": span['attributes']}
                })
        
        # Determine span kind based on name and attributes
        span_kind = 1  # SPAN_KIND_INTERNAL (default)
        if 'user_message' in (span['attributes'] or ''):
            span_kind = 3  # SPAN_KIND_CLIENT
        elif 'tool_name' in (span['attributes'] or ''):
            span_kind = 3  # SPAN_KIND_CLIENT
        
        # Add standard attributes
        attributes.extend([
            {
                "key": "service.name",
                "value": {"stringValue": self.service_name}
            },
            {
                "key": "service.version", 
                "value": {"stringValue": self.service_version}
            },
            {
                "key": "span.original_id",
                "value": {"stringValue": span['span_id']}
            }
        ])
        
        # Convert timestamps
        start_time = self.convert_timestamp_to_nano(span['start_time'])
        end_time = self.convert_timestamp_to_nano(span['end_time'] or span['start_time'])
        
        # Ensure end_time is after start_time
        if end_time <= start_time:
            end_time = start_time + 1_000_000  # Add 1ms
        
        return OTLPSpan(
            trace_id=self.generate_trace_id(trace_id),
            span_id=self.generate_span_id(span['span_id']),
            parent_span_id=self.generate_span_id(span['parent_span_id']) if span['parent_span_id'] else None,
            name=span['name'] or 'unknown_operation',
            kind=span_kind,
            start_time_unix_nano=start_time,
            end_time_unix_nano=end_time,
            attributes=attributes,
            status={"code": 1}  # STATUS_CODE_OK
        )
    
    def convert_trace_to_otlp(self, trace_info: Dict[str, Any]) -> OTLPTrace:
        """Convert a complete trace to OTLP format."""
        trace_id = trace_info['trace_id']
        spans = self.get_spans_for_trace(trace_id)
        
        # Convert all spans
        otlp_spans = []
        for span in spans:
            otlp_span = self.convert_span_to_otlp(span, trace_id)
            
            # Convert to OTLP span format
            span_dict = {
                "traceId": otlp_span.trace_id,
                "spanId": otlp_span.span_id,
                "name": otlp_span.name,
                "kind": otlp_span.kind,
                "startTimeUnixNano": str(otlp_span.start_time_unix_nano),
                "endTimeUnixNano": str(otlp_span.end_time_unix_nano),
                "attributes": otlp_span.attributes,
                "status": otlp_span.status
            }
            
            if otlp_span.parent_span_id:
                span_dict["parentSpanId"] = otlp_span.parent_span_id
            
            otlp_spans.append(span_dict)
        
        # Create resource
        resource = {
            "attributes": [
                {
                    "key": "service.name",
                    "value": {"stringValue": self.service_name}
                },
                {
                    "key": "service.version",
                    "value": {"stringValue": self.service_version}
                },
                {
                    "key": "telemetry.sdk.name",
                    "value": {"stringValue": "llama-stack"}
                },
                {
                    "key": "conversation.trace_id",
                    "value": {"stringValue": trace_id}
                }
            ]
        }
        
        # Create scope spans
        scope_spans = [{
            "scope": {
                "name": "llama-stack-conversation-exporter",
                "version": "1.0.0"
            },
            "spans": otlp_spans
        }]
        
        return OTLPTrace(
            resource=resource,
            scope_spans=scope_spans
        )
    
    def export_to_otlp_json(self, output_path: str, days: Optional[int] = None, limit: Optional[int] = None, compress: bool = False):
        """Export conversations to OTLP JSON format."""
        print(f"🔍 Scanning for conversation traces...")
        
        traces = self.get_conversation_traces(days, limit)
        
        if not traces:
            print("📭 No conversation traces found.")
            return
        
        print(f"📊 Found {len(traces)} conversation traces to export")
        
        # Convert all traces to OTLP format
        resource_spans = []
        
        for i, trace_info in enumerate(traces, 1):
            print(f"🔄 Processing trace {i}/{len(traces)}: {trace_info['trace_id'][:16]}...")
            
            try:
                otlp_trace = self.convert_trace_to_otlp(trace_info)
                
                resource_spans.append({
                    "resource": otlp_trace.resource,
                    "scopeSpans": otlp_trace.scope_spans
                })
                
            except Exception as e:
                print(f"⚠️  Error processing trace {trace_info['trace_id']}: {e}")
                continue
        
        # Create final OTLP export format
        otlp_export = {
            "resourceSpans": resource_spans
        }
        
        # Write to file
        print(f"💾 Writing OTLP export to {output_path}...")
        
        if compress:
            with gzip.open(output_path, 'wt', encoding='utf-8') as f:
                json.dump(otlp_export, f, indent=2)
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(otlp_export, f, indent=2)
        
        print(f"✅ Successfully exported {len(resource_spans)} traces to {output_path}")
        print(f"📈 Total spans exported: {sum(len(rs['scopeSpans'][0]['spans']) for rs in resource_spans)}")
        
        # Show import instructions
        print(f"\n💡 Import Instructions:")
        print(f"   • Jaeger: Use jaeger-query to import OTLP JSON")
        print(f"   • Grafana Tempo: Configure OTLP receiver")
        print(f"   • Custom tools: Parse as standard OTLP JSON format")
    
    def export_to_jaeger_json(self, output_path: str, days: Optional[int] = None, limit: Optional[int] = None):
        """Export conversations to Jaeger-compatible JSON format."""
        print(f"🔍 Scanning for conversation traces (Jaeger format)...")
        
        traces = self.get_conversation_traces(days, limit)
        
        if not traces:
            print("📭 No conversation traces found.")
            return
        
        print(f"📊 Found {len(traces)} conversation traces to export")
        
        jaeger_traces = []
        
        for i, trace_info in enumerate(traces, 1):
            print(f"🔄 Processing trace {i}/{len(traces)}: {trace_info['trace_id'][:16]}...")
            
            try:
                spans = self.get_spans_for_trace(trace_info['trace_id'])
                jaeger_spans = []
                
                for span in spans:
                    # Convert to Jaeger span format
                    jaeger_span = {
                        "traceID": self.generate_trace_id(trace_info['trace_id']),
                        "spanID": self.generate_span_id(span['span_id']),
                        "operationName": span['name'] or 'unknown_operation',
                        "startTime": self.convert_timestamp_to_nano(span['start_time']) // 1000,  # Jaeger uses microseconds
                        "duration": (self.convert_timestamp_to_nano(span['end_time'] or span['start_time']) - 
                                   self.convert_timestamp_to_nano(span['start_time'])) // 1000,
                        "tags": [],
                        "process": {
                            "serviceName": self.service_name,
                            "tags": [
                                {"key": "service.version", "type": "string", "value": self.service_version}
                            ]
                        }
                    }
                    
                    if span['parent_span_id']:
                        jaeger_span["parentSpanID"] = self.generate_span_id(span['parent_span_id'])
                    
                    # Add attributes as tags
                    if span['attributes']:
                        try:
                            attrs = json.loads(span['attributes'])
                            for key, value in attrs.items():
                                jaeger_span["tags"].append({
                                    "key": key,
                                    "type": "string",
                                    "value": str(value)
                                })
                        except json.JSONDecodeError:
                            jaeger_span["tags"].append({
                                "key": "raw_attributes",
                                "type": "string", 
                                "value": span['attributes']
                            })
                    
                    jaeger_spans.append(jaeger_span)
                
                jaeger_traces.append({
                    "traceID": self.generate_trace_id(trace_info['trace_id']),
                    "spans": jaeger_spans,
                    "processes": {
                        "p1": {
                            "serviceName": self.service_name,
                            "tags": [
                                {"key": "service.version", "type": "string", "value": self.service_version}
                            ]
                        }
                    }
                })
                
            except Exception as e:
                print(f"⚠️  Error processing trace {trace_info['trace_id']}: {e}")
                continue
        
        # Write Jaeger format
        jaeger_export = {"data": jaeger_traces}
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(jaeger_export, f, indent=2)
        
        print(f"✅ Successfully exported {len(jaeger_traces)} traces to {output_path} (Jaeger format)")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Export Llama Stack conversations to OpenTelemetry format")
    parser.add_argument(
        "--db",
        default="/Users/azaalouk/.llama/distributions/distribution-myenv-ollama/trace_store.db",
        help="Path to the telemetry database"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output file path"
    )
    parser.add_argument(
        "--format",
        choices=["otlp-json", "jaeger"],
        default="otlp-json",
        help="Export format (default: otlp-json)"
    )
    parser.add_argument(
        "--days",
        type=int,
        help="Only export traces from the last N days"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of conversations to export (for testing)"
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Compress output with gzip (only for otlp-json)"
    )
    
    args = parser.parse_args()
    
    try:
        exporter = ConversationOTLPExporter(args.db)
        
        if args.format == "otlp-json":
            exporter.export_to_otlp_json(args.output, args.days, args.limit, args.compress)
        elif args.format == "jaeger":
            exporter.export_to_jaeger_json(args.output, args.days, args.limit)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
