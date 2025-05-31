# OpenTelemetry Export Validation Guide

This guide shows you how to validate that your exported OTLP files conform to OpenTelemetry specifications and can be imported into other observability tools.

## 🔍 OTLP Validator Tool

The `otel_validator.py` tool validates exported OTLP JSON files against OpenTelemetry specifications to ensure they conform to the standard format.

### Quick Validation
```bash
# Export and validate in one go
python otel_conversation_exporter.py -o test.json --days 1
python otel_validator.py test.json
```

### Validation Features
- ✅ **JSON Structure** - Validates proper OTLP JSON format
- ✅ **Required Fields** - Checks all mandatory OTLP fields are present
- ✅ **Data Types** - Validates field types match OTLP specification
- ✅ **ID Formats** - Ensures trace/span IDs are valid hex strings
- ✅ **Timestamps** - Validates Unix nanosecond timestamps
- ✅ **Relationships** - Checks parent-child span relationships
- ✅ **Attributes** - Validates attribute structure and values
- ✅ **Statistics** - Provides detailed export statistics

## 📋 Validation Command Reference

### Basic Usage
```bash
python otel_validator.py <file>
```

### Options
```bash
python otel_validator.py <file> --strict    # Treat warnings as errors
```

### Examples
```bash
# Validate OTLP export
python otel_validator.py conversations.json

# Validate compressed export
python otel_validator.py conversations.json.gz

# Strict validation (warnings become errors)
python otel_validator.py conversations.json --strict
```

## ✅ Validation Output

### Successful Validation
```
🔍 Validating OTLP file: conversations.json
✅ JSON file loaded successfully

📊 Validation Statistics:
   Resource Spans: 258
   Scope Spans: 258
   Total Spans: 2339
   Unique Traces: 258
   Unique Span IDs: 2346
   Total Attributes: 22115

✅ Validation PASSED - File conforms to OTLP specification!
```

### Validation with Warnings
```
🔍 Validating OTLP file: conversations.json
✅ JSON file loaded successfully

📊 Validation Statistics:
   Resource Spans: 10
   Scope Spans: 10
   Total Spans: 45
   Unique Traces: 10
   Unique Span IDs: 45
   Total Attributes: 180

⚠️  Warnings (2):
   • ResourceSpan[3].scopeSpans[0].scope missing recommended field: 'name'
   • Unexpected root fields: {'extra_field'}

✅ Validation PASSED - File conforms to OTLP specification!
   Note: 2 warnings found (non-critical)
```

### Validation Errors
```
🔍 Validating OTLP file: invalid.json
✅ JSON file loaded successfully

📊 Validation Statistics:
   Resource Spans: 1
   Scope Spans: 1
   Total Spans: 3
   Unique Traces: 1
   Unique Span IDs: 3
   Total Attributes: 12

❌ Errors (3):
   • ResourceSpan[0].scopeSpans[0].spans[1] missing required field: 'traceId'
   • ResourceSpan[0].scopeSpans[0].spans[1].spanId must be 16 characters (got 12)
   • ResourceSpan[0].scopeSpans[0].spans[2] endTimeUnixNano must be >= startTimeUnixNano

🚫 Validation FAILED - File does not conform to OTLP specification
```

## 🔧 What Gets Validated

### Root Structure
- ✅ `resourceSpans` array is present and valid
- ⚠️ No unexpected root-level fields

### Resource Spans
- ✅ Each resource span is a valid object
- ✅ `scopeSpans` array is present
- ✅ Resource attributes follow OTLP format

### Scope Spans
- ✅ Each scope span is a valid object
- ✅ `spans` array is present
- ⚠️ Instrumentation scope has recommended `name` field

### Spans (Core Validation)
- ✅ **Required fields**: `traceId`, `spanId`, `name`, `startTimeUnixNano`, `endTimeUnixNano`
- ✅ **Trace ID**: 32-character hexadecimal string
- ✅ **Span ID**: 16-character hexadecimal string
- ✅ **Parent Span ID**: Valid 16-character hex (if present)
- ✅ **Timestamps**: Valid Unix nanosecond timestamps
- ✅ **Timing**: End time >= start time
- ✅ **Span Kind**: Valid integer (0-5) if present
- ✅ **Status**: Valid status code (0-2) if present

### Attributes
- ✅ Attributes array structure
- ✅ Each attribute has `key` and `value` fields
- ✅ Keys are strings
- ✅ Values have exactly one type field
- ✅ Value types match OTLP specification:
  - `stringValue` (string)
  - `boolValue` (boolean)
  - `intValue` (integer/string)
  - `doubleValue` (number)
  - `arrayValue` (array)
  - `kvlistValue` (key-value list)
  - `bytesValue` (bytes)

## 🎯 Common Validation Issues

### Invalid Trace/Span IDs
```
❌ ResourceSpan[0].scopeSpans[0].spans[1].traceId must be 32 characters (got 28)
❌ ResourceSpan[0].scopeSpans[0].spans[2].spanId must be a valid hexadecimal string
```
**Fix**: Ensure IDs are properly formatted hex strings of correct length

### Missing Required Fields
```
❌ ResourceSpan[0].scopeSpans[0].spans[1] missing required field: 'name'
❌ ResourceSpan[0].scopeSpans[0] missing required field: 'spans'
```
**Fix**: Add all required OTLP fields to your export

### Invalid Timestamps
```
❌ ResourceSpan[0].scopeSpans[0].spans[1].startTimeUnixNano must be a valid integer string
❌ ResourceSpan[0].scopeSpans[0].spans[2] endTimeUnixNano must be >= startTimeUnixNano
```
**Fix**: Use valid Unix nanosecond timestamps with proper ordering

### Malformed Attributes
```
❌ ResourceSpan[0].scopeSpans[0].spans[1].attributes[0] missing required field: 'key'
❌ ResourceSpan[0].scopeSpans[0].spans[1].attributes[1].value must have exactly one value type
```
**Fix**: Ensure attributes follow OTLP attribute specification

## 🚀 Integration Testing

### Test Import Compatibility
```bash
# 1. Export and validate
python otel_conversation_exporter.py -o test.json --days 1
python otel_validator.py test.json

# 2. Test with your target tool
# For Jaeger:
python otel_conversation_exporter.py -o jaeger_test.json --format jaeger
# Import into Jaeger and verify

# For Grafana Tempo:
# Configure Tempo OTLP receiver and test import
```

### Automated Validation
```bash
#!/bin/bash
# Validation script for CI/CD
export_file="conversations_$(date +%Y%m%d).json"
python otel_conversation_exporter.py -o "$export_file" --days 7

if python otel_validator.py "$export_file"; then
    echo "✅ Export validation passed"
    # Upload to observability platform
else
    echo "❌ Export validation failed"
    exit 1
fi
```

### Validation in Python
```python
#!/usr/bin/env python3
import subprocess
import sys

def validate_export(file_path):
    """Validate OTLP export programmatically."""
    result = subprocess.run([
        'python', 'otel_validator.py', file_path
    ], capture_output=True, text=True)
    
    return result.returncode == 0, result.stdout, result.stderr

# Usage
is_valid, stdout, stderr = validate_export('conversations.json')
if is_valid:
    print("✅ Validation passed")
else:
    print("❌ Validation failed")
    print(stderr)
```

## 🔍 Validation Best Practices

### Regular Validation
- ✅ Validate exports before importing into production systems
- ✅ Include validation in automated export scripts
- ✅ Test with small exports first
- ✅ Validate after any exporter changes

### Troubleshooting
- 🔧 Use `--strict` mode to catch potential issues early
- 🔧 Check validation statistics for data consistency
- 🔧 Test imports with target observability tools
- 🔧 Validate compressed and uncompressed exports

### Quality Assurance
- 📊 Monitor validation statistics over time
- 📊 Track unique trace/span counts
- 📊 Verify attribute counts match expectations
- 📊 Ensure timestamp ranges are reasonable

The OTLP validator ensures your Llama Stack conversation exports are fully compatible with the OpenTelemetry ecosystem and can be reliably imported into any OTLP-compatible observability tool.
