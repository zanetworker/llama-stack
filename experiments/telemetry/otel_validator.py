#!/usr/bin/env python3
"""
OpenTelemetry Format Validator for Llama Stack Exports

This tool validates exported OTLP JSON files against OpenTelemetry specifications
to ensure they conform to the standard format and can be imported into OTEL tools.
"""

import argparse
import json
import sys
import gzip
from typing import Any, Dict, List, Optional, Set
from datetime import datetime


class OTLPValidator:
    """Validates OTLP JSON exports against OpenTelemetry specifications."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.stats = {
            'resource_spans': 0,
            'scope_spans': 0,
            'spans': 0,
            'traces': 0,
            'attributes': 0
        }
        self.trace_ids = set()
        self.span_ids = set()
    
    def validate_file(self, file_path: str) -> bool:
        """Validate an OTLP JSON file."""
        print(f"🔍 Validating OTLP file: {file_path}")
        
        try:
            # Load JSON data
            if file_path.endswith('.gz'):
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            print(f"✅ JSON file loaded successfully")
            
            # Validate root structure
            self._validate_root_structure(data)
            
            # Validate resource spans
            if 'resourceSpans' in data:
                self._validate_resource_spans(data['resourceSpans'])
            
            # Print results
            self._print_validation_results()
            
            return len(self.errors) == 0
            
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON format: {e}")
            self._print_validation_results()
            return False
        except Exception as e:
            self.errors.append(f"Validation error: {e}")
            self._print_validation_results()
            return False
    
    def _validate_root_structure(self, data: Dict[str, Any]):
        """Validate the root OTLP structure."""
        if not isinstance(data, dict):
            self.errors.append("Root element must be a JSON object")
            return
        
        # Check required fields
        if 'resourceSpans' not in data:
            self.errors.append("Missing required field: 'resourceSpans'")
            return
        
        if not isinstance(data['resourceSpans'], list):
            self.errors.append("'resourceSpans' must be an array")
            return
        
        self.stats['resource_spans'] = len(data['resourceSpans'])
        
        # Check for unexpected fields at root level
        expected_fields = {'resourceSpans'}
        actual_fields = set(data.keys())
        unexpected = actual_fields - expected_fields
        if unexpected:
            self.warnings.append(f"Unexpected root fields: {unexpected}")
    
    def _validate_resource_spans(self, resource_spans: List[Dict[str, Any]]):
        """Validate resource spans array."""
        for i, resource_span in enumerate(resource_spans):
            self._validate_resource_span(resource_span, i)
    
    def _validate_resource_span(self, resource_span: Dict[str, Any], index: int):
        """Validate a single resource span."""
        if not isinstance(resource_span, dict):
            self.errors.append(f"ResourceSpan[{index}] must be an object")
            return
        
        # Validate resource
        if 'resource' in resource_span:
            self._validate_resource(resource_span['resource'], index)
        
        # Validate scope spans (required)
        if 'scopeSpans' not in resource_span:
            self.errors.append(f"ResourceSpan[{index}] missing required field: 'scopeSpans'")
            return
        
        if not isinstance(resource_span['scopeSpans'], list):
            self.errors.append(f"ResourceSpan[{index}].scopeSpans must be an array")
            return
        
        self.stats['scope_spans'] += len(resource_span['scopeSpans'])
        
        for j, scope_span in enumerate(resource_span['scopeSpans']):
            self._validate_scope_span(scope_span, index, j)
    
    def _validate_resource(self, resource: Dict[str, Any], rs_index: int):
        """Validate resource object."""
        if not isinstance(resource, dict):
            self.errors.append(f"ResourceSpan[{rs_index}].resource must be an object")
            return
        
        # Validate attributes if present
        if 'attributes' in resource:
            self._validate_attributes(resource['attributes'], f"ResourceSpan[{rs_index}].resource")
    
    def _validate_scope_span(self, scope_span: Dict[str, Any], rs_index: int, ss_index: int):
        """Validate a scope span."""
        if not isinstance(scope_span, dict):
            self.errors.append(f"ResourceSpan[{rs_index}].scopeSpans[{ss_index}] must be an object")
            return
        
        # Validate scope (optional but recommended)
        if 'scope' in scope_span:
            self._validate_scope(scope_span['scope'], rs_index, ss_index)
        
        # Validate spans (required)
        if 'spans' not in scope_span:
            self.errors.append(f"ResourceSpan[{rs_index}].scopeSpans[{ss_index}] missing required field: 'spans'")
            return
        
        if not isinstance(scope_span['spans'], list):
            self.errors.append(f"ResourceSpan[{rs_index}].scopeSpans[{ss_index}].spans must be an array")
            return
        
        self.stats['spans'] += len(scope_span['spans'])
        
        for k, span in enumerate(scope_span['spans']):
            self._validate_span(span, rs_index, ss_index, k)
    
    def _validate_scope(self, scope: Dict[str, Any], rs_index: int, ss_index: int):
        """Validate instrumentation scope."""
        if not isinstance(scope, dict):
            self.errors.append(f"ResourceSpan[{rs_index}].scopeSpans[{ss_index}].scope must be an object")
            return
        
        # Name is recommended
        if 'name' not in scope:
            self.warnings.append(f"ResourceSpan[{rs_index}].scopeSpans[{ss_index}].scope missing recommended field: 'name'")
        elif not isinstance(scope['name'], str):
            self.errors.append(f"ResourceSpan[{rs_index}].scopeSpans[{ss_index}].scope.name must be a string")
        
        # Version is optional
        if 'version' in scope and not isinstance(scope['version'], str):
            self.errors.append(f"ResourceSpan[{rs_index}].scopeSpans[{ss_index}].scope.version must be a string")
    
    def _validate_span(self, span: Dict[str, Any], rs_index: int, ss_index: int, s_index: int):
        """Validate a single span."""
        span_path = f"ResourceSpan[{rs_index}].scopeSpans[{ss_index}].spans[{s_index}]"
        
        if not isinstance(span, dict):
            self.errors.append(f"{span_path} must be an object")
            return
        
        # Required fields
        required_fields = ['traceId', 'spanId', 'name', 'startTimeUnixNano', 'endTimeUnixNano']
        for field in required_fields:
            if field not in span:
                self.errors.append(f"{span_path} missing required field: '{field}'")
        
        # Validate trace ID
        if 'traceId' in span:
            self._validate_trace_id(span['traceId'], span_path)
        
        # Validate span ID
        if 'spanId' in span:
            self._validate_span_id(span['spanId'], span_path)
        
        # Validate parent span ID (optional)
        if 'parentSpanId' in span:
            self._validate_span_id(span['parentSpanId'], f"{span_path}.parentSpanId")
        
        # Validate name
        if 'name' in span and not isinstance(span['name'], str):
            self.errors.append(f"{span_path}.name must be a string")
        
        # Validate timestamps
        if 'startTimeUnixNano' in span:
            self._validate_timestamp(span['startTimeUnixNano'], f"{span_path}.startTimeUnixNano")
        
        if 'endTimeUnixNano' in span:
            self._validate_timestamp(span['endTimeUnixNano'], f"{span_path}.endTimeUnixNano")
        
        # Validate timing relationship
        if 'startTimeUnixNano' in span and 'endTimeUnixNano' in span:
            try:
                start = int(span['startTimeUnixNano'])
                end = int(span['endTimeUnixNano'])
                if end < start:
                    self.errors.append(f"{span_path} endTimeUnixNano must be >= startTimeUnixNano")
            except (ValueError, TypeError):
                pass  # Already caught by timestamp validation
        
        # Validate kind (optional)
        if 'kind' in span:
            self._validate_span_kind(span['kind'], span_path)
        
        # Validate attributes (optional)
        if 'attributes' in span:
            self._validate_attributes(span['attributes'], span_path)
        
        # Validate status (optional)
        if 'status' in span:
            self._validate_status(span['status'], span_path)
    
    def _validate_trace_id(self, trace_id: Any, path: str):
        """Validate trace ID format."""
        if not isinstance(trace_id, str):
            self.errors.append(f"{path}.traceId must be a string")
            return
        
        if len(trace_id) != 32:
            self.errors.append(f"{path}.traceId must be 32 characters (got {len(trace_id)})")
            return
        
        try:
            int(trace_id, 16)  # Must be valid hex
            self.trace_ids.add(trace_id)
        except ValueError:
            self.errors.append(f"{path}.traceId must be a valid hexadecimal string")
    
    def _validate_span_id(self, span_id: Any, path: str):
        """Validate span ID format."""
        if not isinstance(span_id, str):
            self.errors.append(f"{path} must be a string")
            return
        
        if len(span_id) != 16:
            self.errors.append(f"{path} must be 16 characters (got {len(span_id)})")
            return
        
        try:
            int(span_id, 16)  # Must be valid hex
            self.span_ids.add(span_id)
        except ValueError:
            self.errors.append(f"{path} must be a valid hexadecimal string")
    
    def _validate_timestamp(self, timestamp: Any, path: str):
        """Validate Unix nano timestamp."""
        if isinstance(timestamp, str):
            try:
                int(timestamp)
            except ValueError:
                self.errors.append(f"{path} must be a valid integer string")
        elif isinstance(timestamp, int):
            pass  # Valid
        else:
            self.errors.append(f"{path} must be a string or integer")
    
    def _validate_span_kind(self, kind: Any, path: str):
        """Validate span kind."""
        if not isinstance(kind, int):
            self.errors.append(f"{path}.kind must be an integer")
            return
        
        # Valid span kinds: 0=UNSPECIFIED, 1=INTERNAL, 2=SERVER, 3=CLIENT, 4=PRODUCER, 5=CONSUMER
        if kind not in [0, 1, 2, 3, 4, 5]:
            self.errors.append(f"{path}.kind must be 0-5 (got {kind})")
    
    def _validate_attributes(self, attributes: Any, path: str):
        """Validate attributes array."""
        if not isinstance(attributes, list):
            self.errors.append(f"{path}.attributes must be an array")
            return
        
        self.stats['attributes'] += len(attributes)
        
        for i, attr in enumerate(attributes):
            self._validate_attribute(attr, f"{path}.attributes[{i}]")
    
    def _validate_attribute(self, attribute: Any, path: str):
        """Validate a single attribute."""
        if not isinstance(attribute, dict):
            self.errors.append(f"{path} must be an object")
            return
        
        # Required fields
        if 'key' not in attribute:
            self.errors.append(f"{path} missing required field: 'key'")
        elif not isinstance(attribute['key'], str):
            self.errors.append(f"{path}.key must be a string")
        
        if 'value' not in attribute:
            self.errors.append(f"{path} missing required field: 'value'")
        else:
            self._validate_attribute_value(attribute['value'], f"{path}.value")
    
    def _validate_attribute_value(self, value: Any, path: str):
        """Validate attribute value."""
        if not isinstance(value, dict):
            self.errors.append(f"{path} must be an object")
            return
        
        # Must have exactly one value type
        value_types = ['stringValue', 'boolValue', 'intValue', 'doubleValue', 'arrayValue', 'kvlistValue', 'bytesValue']
        present_types = [vt for vt in value_types if vt in value]
        
        if len(present_types) == 0:
            self.errors.append(f"{path} must have one value type: {value_types}")
        elif len(present_types) > 1:
            self.errors.append(f"{path} must have exactly one value type (found: {present_types})")
        
        # Validate specific value types
        if 'stringValue' in value and not isinstance(value['stringValue'], str):
            self.errors.append(f"{path}.stringValue must be a string")
        
        if 'boolValue' in value and not isinstance(value['boolValue'], bool):
            self.errors.append(f"{path}.boolValue must be a boolean")
        
        if 'intValue' in value and not isinstance(value['intValue'], (int, str)):
            self.errors.append(f"{path}.intValue must be an integer or string")
        
        if 'doubleValue' in value and not isinstance(value['doubleValue'], (int, float)):
            self.errors.append(f"{path}.doubleValue must be a number")
    
    def _validate_status(self, status: Any, path: str):
        """Validate span status."""
        if not isinstance(status, dict):
            self.errors.append(f"{path}.status must be an object")
            return
        
        # Validate code (optional but recommended)
        if 'code' in status:
            if not isinstance(status['code'], int):
                self.errors.append(f"{path}.status.code must be an integer")
            elif status['code'] not in [0, 1, 2]:  # UNSET=0, OK=1, ERROR=2
                self.errors.append(f"{path}.status.code must be 0, 1, or 2 (got {status['code']})")
        
        # Validate message (optional)
        if 'message' in status and not isinstance(status['message'], str):
            self.errors.append(f"{path}.status.message must be a string")
    
    def _print_validation_results(self):
        """Print validation results."""
        print(f"\n📊 Validation Statistics:")
        print(f"   Resource Spans: {self.stats['resource_spans']}")
        print(f"   Scope Spans: {self.stats['scope_spans']}")
        print(f"   Total Spans: {self.stats['spans']}")
        print(f"   Unique Traces: {len(self.trace_ids)}")
        print(f"   Unique Span IDs: {len(self.span_ids)}")
        print(f"   Total Attributes: {self.stats['attributes']}")
        
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"   • {error}")
            print(f"\n🚫 Validation FAILED - File does not conform to OTLP specification")
        else:
            print(f"\n✅ Validation PASSED - File conforms to OTLP specification!")
            if self.warnings:
                print(f"   Note: {len(self.warnings)} warnings found (non-critical)")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Validate OTLP JSON exports against OpenTelemetry specifications")
    parser.add_argument(
        "file",
        help="Path to the OTLP JSON file to validate"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors"
    )
    
    args = parser.parse_args()
    
    try:
        validator = OTLPValidator()
        is_valid = validator.validate_file(args.file)
        
        if args.strict and validator.warnings:
            print(f"\n🚫 Strict mode: Treating {len(validator.warnings)} warnings as errors")
            is_valid = False
        
        sys.exit(0 if is_valid else 1)
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
