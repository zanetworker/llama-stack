#!/usr/bin/env python3
"""
Conversation Patterns Analyzer for Llama Stack Telemetry

This tool analyzes conversation patterns, user behavior, and system performance
to provide insights for improving your LLM/agent interactions.
"""

import argparse
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, Counter
import statistics


class ConversationPatternsAnalyzer:
    """Analyzes patterns in conversation data to provide insights."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def analyze_usage_patterns(self, days_back: int = 30) -> Dict[str, Any]:
        """Analyze usage patterns over time."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
        
        # Get all traces with timestamps
        query = """
        SELECT trace_id, start_time, end_time
        FROM traces 
        WHERE start_time > ?
        ORDER BY start_time
        """
        
        cursor.execute(query, (cutoff_date,))
        traces = cursor.fetchall()
        
        # Analyze by hour, day of week, and date
        hourly_usage = defaultdict(int)
        daily_usage = defaultdict(int)
        weekly_usage = defaultdict(int)
        
        for trace in traces:
            try:
                dt = datetime.fromisoformat(trace['start_time'].replace('Z', '+00:00'))
                
                # Hour of day (0-23)
                hourly_usage[dt.hour] += 1
                
                # Day of week (0=Monday, 6=Sunday)
                weekly_usage[dt.weekday()] += 1
                
                # Date
                date_str = dt.strftime('%Y-%m-%d')
                daily_usage[date_str] += 1
                
            except:
                continue
        
        conn.close()
        
        # Find peak times
        peak_hour = max(hourly_usage.items(), key=lambda x: x[1]) if hourly_usage else (0, 0)
        peak_day = max(weekly_usage.items(), key=lambda x: x[1]) if weekly_usage else (0, 0)
        
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        return {
            'total_traces': len(traces),
            'hourly_distribution': dict(hourly_usage),
            'weekly_distribution': dict(weekly_usage),
            'daily_usage': dict(daily_usage),
            'peak_hour': peak_hour[0],
            'peak_hour_count': peak_hour[1],
            'peak_day': day_names[peak_day[0]] if peak_day[0] < 7 else 'Unknown',
            'peak_day_count': peak_day[1],
            'avg_daily_usage': sum(daily_usage.values()) / len(daily_usage) if daily_usage else 0
        }
    
    def analyze_performance_patterns(self, days_back: int = 30) -> Dict[str, Any]:
        """Analyze performance patterns and bottlenecks."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
        
        # Get spans with timing data
        query = """
        SELECT s.name, s.start_time, s.end_time, s.status, s.attributes
        FROM spans s
        JOIN traces t ON s.trace_id = t.trace_id
        WHERE t.start_time > ? AND s.start_time IS NOT NULL AND s.end_time IS NOT NULL
        """
        
        cursor.execute(query, (cutoff_date,))
        spans = cursor.fetchall()
        
        # Analyze performance by operation type
        operation_times = defaultdict(list)
        error_counts = defaultdict(int)
        total_operations = defaultdict(int)
        
        for span in spans:
            try:
                start = datetime.fromisoformat(span['start_time'].replace('Z', '+00:00'))
                end = datetime.fromisoformat(span['end_time'].replace('Z', '+00:00'))
                duration_ms = (end - start).total_seconds() * 1000
                
                operation_name = span['name']
                operation_times[operation_name].append(duration_ms)
                total_operations[operation_name] += 1
                
                if span['status'] == 'error':
                    error_counts[operation_name] += 1
                    
            except:
                continue
        
        # Calculate statistics for each operation
        performance_stats = {}
        for operation, times in operation_times.items():
            if times:
                performance_stats[operation] = {
                    'count': len(times),
                    'avg_duration_ms': statistics.mean(times),
                    'median_duration_ms': statistics.median(times),
                    'min_duration_ms': min(times),
                    'max_duration_ms': max(times),
                    'std_duration_ms': statistics.stdev(times) if len(times) > 1 else 0,
                    'error_count': error_counts[operation],
                    'error_rate': error_counts[operation] / total_operations[operation] if total_operations[operation] > 0 else 0
                }
        
        # Find slowest operations
        slowest_ops = sorted(
            performance_stats.items(),
            key=lambda x: x[1]['avg_duration_ms'],
            reverse=True
        )[:10]
        
        # Find operations with highest error rates
        error_prone_ops = sorted(
            performance_stats.items(),
            key=lambda x: x[1]['error_rate'],
            reverse=True
        )[:10]
        
        conn.close()
        
        return {
            'total_spans_analyzed': len(spans),
            'performance_by_operation': performance_stats,
            'slowest_operations': slowest_ops,
            'error_prone_operations': error_prone_ops,
            'overall_error_rate': sum(error_counts.values()) / sum(total_operations.values()) if total_operations else 0
        }
    
    def analyze_conversation_content(self, days_back: int = 30) -> Dict[str, Any]:
        """Analyze conversation content patterns."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
        
        # Get spans with message content
        query = """
        SELECT s.attributes
        FROM spans s
        JOIN traces t ON s.trace_id = t.trace_id
        WHERE t.start_time > ? AND s.attributes IS NOT NULL
        """
        
        cursor.execute(query, (cutoff_date,))
        spans = cursor.fetchall()
        
        user_messages = []
        assistant_messages = []
        tool_usage = defaultdict(int)
        message_lengths = []
        
        for span in spans:
            try:
                attrs = json.loads(span['attributes'])
                
                if 'user_message' in attrs:
                    message = str(attrs['user_message'])
                    user_messages.append(message)
                    message_lengths.append(len(message))
                
                if 'assistant_message' in attrs:
                    message = str(attrs['assistant_message'])
                    assistant_messages.append(message)
                    message_lengths.append(len(message))
                
                if 'tool_name' in attrs:
                    tool_usage[attrs['tool_name']] += 1
                    
            except:
                continue
        
        # Analyze message patterns
        common_user_phrases = self._extract_common_phrases(user_messages)
        common_assistant_phrases = self._extract_common_phrases(assistant_messages)
        
        # Message length statistics
        length_stats = {}
        if message_lengths:
            length_stats = {
                'avg_length': statistics.mean(message_lengths),
                'median_length': statistics.median(message_lengths),
                'min_length': min(message_lengths),
                'max_length': max(message_lengths)
            }
        
        conn.close()
        
        return {
            'total_user_messages': len(user_messages),
            'total_assistant_messages': len(assistant_messages),
            'tool_usage_frequency': dict(tool_usage),
            'most_used_tools': sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)[:10],
            'common_user_phrases': common_user_phrases[:20],
            'common_assistant_phrases': common_assistant_phrases[:20],
            'message_length_stats': length_stats
        }
    
    def _extract_common_phrases(self, messages: List[str]) -> List[Tuple[str, int]]:
        """Extract common phrases from messages."""
        if not messages:
            return []
        
        # Simple phrase extraction - split by common delimiters and count
        phrase_counts = Counter()
        
        for message in messages:
            # Split into potential phrases
            words = message.lower().split()
            
            # Extract 2-3 word phrases
            for i in range(len(words) - 1):
                if i < len(words) - 2:
                    phrase = ' '.join(words[i:i+3])
                    if len(phrase) > 10:  # Filter out very short phrases
                        phrase_counts[phrase] += 1
                
                phrase = ' '.join(words[i:i+2])
                if len(phrase) > 6:
                    phrase_counts[phrase] += 1
        
        return phrase_counts.most_common()
    
    def generate_insights(self, usage_patterns: Dict, performance_patterns: Dict, content_patterns: Dict) -> List[str]:
        """Generate actionable insights from the analysis."""
        insights = []
        
        # Usage insights
        if usage_patterns['total_traces'] > 0:
            insights.append(f"📈 System processed {usage_patterns['total_traces']} conversations in the analyzed period")
            
            if usage_patterns['peak_hour_count'] > 0:
                insights.append(f"⏰ Peak usage occurs at {usage_patterns['peak_hour']:02d}:00 with {usage_patterns['peak_hour_count']} conversations")
            
            if usage_patterns['peak_day_count'] > 0:
                insights.append(f"📅 Busiest day is {usage_patterns['peak_day']} with {usage_patterns['peak_day_count']} conversations")
        
        # Performance insights
        if performance_patterns['slowest_operations']:
            slowest_op = performance_patterns['slowest_operations'][0]
            insights.append(f"🐌 Slowest operation: '{slowest_op[0]}' averaging {slowest_op[1]['avg_duration_ms']:.0f}ms")
        
        if performance_patterns['overall_error_rate'] > 0:
            error_rate = performance_patterns['overall_error_rate'] * 100
            insights.append(f"⚠️  Overall error rate: {error_rate:.1f}%")
            
            if performance_patterns['error_prone_operations']:
                error_op = performance_patterns['error_prone_operations'][0]
                if error_op[1]['error_rate'] > 0.1:  # More than 10% error rate
                    insights.append(f"🚨 High error rate in '{error_op[0]}': {error_op[1]['error_rate']*100:.1f}%")
        
        # Content insights
        if content_patterns['total_user_messages'] > 0:
            avg_ratio = content_patterns['total_assistant_messages'] / content_patterns['total_user_messages']
            insights.append(f"💬 Assistant responds {avg_ratio:.1f} times per user message on average")
        
        if content_patterns['most_used_tools']:
            top_tool = content_patterns['most_used_tools'][0]
            insights.append(f"🔧 Most used tool: '{top_tool[0]}' ({top_tool[1]} times)")
        
        if content_patterns['message_length_stats']:
            avg_len = content_patterns['message_length_stats']['avg_length']
            insights.append(f"📝 Average message length: {avg_len:.0f} characters")
        
        # Recommendations
        insights.append("\n🎯 Recommendations:")
        
        if performance_patterns['slowest_operations']:
            insights.append("• Consider optimizing the slowest operations for better user experience")
        
        if performance_patterns['overall_error_rate'] > 0.05:  # More than 5% error rate
            insights.append("• Investigate and fix high error rates to improve reliability")
        
        if usage_patterns['peak_hour_count'] > usage_patterns['avg_daily_usage'] * 2:
            insights.append("• Consider scaling resources during peak usage hours")
        
        if content_patterns['tool_usage_frequency']:
            insights.append("• Tool usage indicates users are engaging with advanced features")
        
        return insights
    
    def print_comprehensive_report(self, days_back: int = 30):
        """Print a comprehensive analysis report."""
        print(f"\n🔍 Comprehensive Conversation Analysis Report")
        print(f"📅 Analyzing data from the last {days_back} days")
        print("=" * 70)
        
        # Gather all analysis data
        usage_patterns = self.analyze_usage_patterns(days_back)
        performance_patterns = self.analyze_performance_patterns(days_back)
        content_patterns = self.analyze_conversation_content(days_back)
        
        # Usage Patterns Section
        print(f"\n📊 Usage Patterns")
        print("-" * 30)
        print(f"Total conversations: {usage_patterns['total_traces']}")
        print(f"Average daily usage: {usage_patterns['avg_daily_usage']:.1f}")
        print(f"Peak hour: {usage_patterns['peak_hour']:02d}:00 ({usage_patterns['peak_hour_count']} conversations)")
        print(f"Peak day: {usage_patterns['peak_day']} ({usage_patterns['peak_day_count']} conversations)")
        
        if usage_patterns['hourly_distribution']:
            print(f"\n⏰ Hourly Distribution (top 5):")
            sorted_hours = sorted(usage_patterns['hourly_distribution'].items(), key=lambda x: x[1], reverse=True)
            for hour, count in sorted_hours[:5]:
                print(f"  {hour:02d}:00 - {count} conversations")
        
        # Performance Patterns Section
        print(f"\n⚡ Performance Analysis")
        print("-" * 30)
        print(f"Total operations analyzed: {performance_patterns['total_spans_analyzed']}")
        print(f"Overall error rate: {performance_patterns['overall_error_rate']*100:.1f}%")
        
        if performance_patterns['slowest_operations']:
            print(f"\n🐌 Slowest Operations:")
            for op_name, stats in performance_patterns['slowest_operations'][:5]:
                print(f"  {op_name}: {stats['avg_duration_ms']:.0f}ms avg ({stats['count']} calls)")
        
        if performance_patterns['error_prone_operations']:
            error_ops = [op for op in performance_patterns['error_prone_operations'] if op[1]['error_rate'] > 0]
            if error_ops:
                print(f"\n⚠️  Operations with Errors:")
                for op_name, stats in error_ops[:5]:
                    print(f"  {op_name}: {stats['error_rate']*100:.1f}% error rate ({stats['error_count']} errors)")
        
        # Content Patterns Section
        print(f"\n💬 Content Analysis")
        print("-" * 30)
        print(f"User messages: {content_patterns['total_user_messages']}")
        print(f"Assistant messages: {content_patterns['total_assistant_messages']}")
        
        if content_patterns['message_length_stats']:
            stats = content_patterns['message_length_stats']
            print(f"Average message length: {stats['avg_length']:.0f} characters")
            print(f"Message length range: {stats['min_length']} - {stats['max_length']} characters")
        
        if content_patterns['most_used_tools']:
            print(f"\n🔧 Tool Usage:")
            for tool_name, count in content_patterns['most_used_tools'][:5]:
                print(f"  {tool_name}: {count} times")
        
        # Insights Section
        insights = self.generate_insights(usage_patterns, performance_patterns, content_patterns)
        print(f"\n💡 Key Insights & Recommendations")
        print("-" * 40)
        for insight in insights:
            print(insight)


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Analyze Llama Stack conversation patterns")
    parser.add_argument(
        "--db",
        default="/Users/azaalouk/.llama/distributions/distribution-myenv-ollama/trace_store.db",
        help="Path to the telemetry database"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Full report command
    report_parser = subparsers.add_parser("report", help="Generate comprehensive analysis report")
    report_parser.add_argument("--days", type=int, default=30, help="Number of days to analyze")
    
    # Usage patterns command
    usage_parser = subparsers.add_parser("usage", help="Analyze usage patterns")
    usage_parser.add_argument("--days", type=int, default=30, help="Number of days to analyze")
    
    # Performance command
    perf_parser = subparsers.add_parser("performance", help="Analyze performance patterns")
    perf_parser.add_argument("--days", type=int, default=30, help="Number of days to analyze")
    
    # Content command
    content_parser = subparsers.add_parser("content", help="Analyze conversation content")
    content_parser.add_argument("--days", type=int, default=30, help="Number of days to analyze")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        analyzer = ConversationPatternsAnalyzer(args.db)
        
        if args.command == "report":
            analyzer.print_comprehensive_report(args.days)
        
        elif args.command == "usage":
            patterns = analyzer.analyze_usage_patterns(args.days)
            print(json.dumps(patterns, indent=2))
        
        elif args.command == "performance":
            patterns = analyzer.analyze_performance_patterns(args.days)
            print(json.dumps(patterns, indent=2, default=str))
        
        elif args.command == "content":
            patterns = analyzer.analyze_conversation_content(args.days)
            print(json.dumps(patterns, indent=2, default=str))
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
