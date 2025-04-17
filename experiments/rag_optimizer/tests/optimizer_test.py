#!/usr/bin/env python3
"""
RAG Parameter Optimizer Tests

This script contains tests for the RAG Parameter Optimizer.
It verifies that the optimizer returns the expected parameters for different use cases.
"""

import sys
import os
import unittest
from typing import Dict, Any

# Add the parent directory to the path to import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from experiments.rag_optimizer.core.base_optimizer import get_rag_parameters as get_base_rag_parameters
from experiments.rag_optimizer.core.enhanced_optimizer import get_rag_parameters as get_enhanced_rag_parameters

class TestBaseOptimizer(unittest.TestCase):
    """Tests for the base RAG Parameter Optimizer."""
    
    def test_known_use_case(self):
        """Test that the optimizer returns the expected parameters for a known use case."""
        params = get_base_rag_parameters("Knowledge Management")
        self.assertEqual(params['use_case'], "Knowledge Management")
        self.assertIn('chunk_size_tokens', params)
        self.assertIn('embedding_model_recommendation', params)
        self.assertIn('top_k', params)
    
    def test_unknown_use_case(self):
        """Test that the optimizer returns base defaults for an unknown use case."""
        params = get_base_rag_parameters("Unknown Use Case")
        self.assertEqual(params['use_case'], "Default")
        self.assertIn('notes', params)
        self.assertIn('requested_use_case', params)
        self.assertEqual(params['requested_use_case'], "Unknown Use Case")
    
    def test_case_insensitivity(self):
        """Test that the optimizer is case-insensitive for use case names."""
        params1 = get_base_rag_parameters("Knowledge Management")
        params2 = get_base_rag_parameters("knowledge management")
        params3 = get_base_rag_parameters("KNOWLEDGE MANAGEMENT")
        
        self.assertEqual(params1['use_case'], params2['use_case'])
        self.assertEqual(params1['use_case'], params3['use_case'])
    
    def test_whitespace_handling(self):
        """Test that the optimizer handles whitespace in use case names."""
        params1 = get_base_rag_parameters("Knowledge Management")
        params2 = get_base_rag_parameters("  Knowledge Management  ")
        
        self.assertEqual(params1['use_case'], params2['use_case'])

class TestEnhancedOptimizer(unittest.TestCase):
    """Tests for the enhanced RAG Parameter Optimizer."""
    
    def test_document_type_adjustment(self):
        """Test that the optimizer adjusts parameters based on document type."""
        base_params = get_enhanced_rag_parameters("Knowledge Management")
        technical_params = get_enhanced_rag_parameters("Knowledge Management", document_type="technical")
        
        self.assertNotEqual(
            base_params['chunk_size_tokens'],
            technical_params['chunk_size_tokens'],
            "Parameters should be adjusted for technical documents"
        )
    
    def test_performance_priority_adjustment(self):
        """Test that the optimizer adjusts parameters based on performance priority."""
        base_params = get_enhanced_rag_parameters("Knowledge Management")
        accuracy_params = get_enhanced_rag_parameters("Knowledge Management", performance_priority="accuracy")
        latency_params = get_enhanced_rag_parameters("Knowledge Management", performance_priority="latency")
        
        self.assertNotEqual(
            accuracy_params['embedding_model_recommendation'],
            latency_params['embedding_model_recommendation'],
            "Parameters should be adjusted differently for accuracy vs. latency"
        )
    
    def test_data_size_adjustment(self):
        """Test that the optimizer adjusts parameters based on data size."""
        base_params = get_enhanced_rag_parameters("Knowledge Management")
        small_params = get_enhanced_rag_parameters("Knowledge Management", data_size="small")
        large_params = get_enhanced_rag_parameters("Knowledge Management", data_size="large")
        
        # Check if top_k is adjusted based on data size
        if isinstance(small_params['top_k'], (int, float)) and isinstance(large_params['top_k'], (int, float)):
            self.assertLess(
                small_params['top_k'],
                large_params['top_k'],
                "Top-k should be smaller for small data and larger for large data"
            )
    
    def test_adjustments_metadata(self):
        """Test that the optimizer includes metadata about the adjustments."""
        params = get_enhanced_rag_parameters(
            "Knowledge Management",
            document_type="technical",
            performance_priority="accuracy",
            data_size="large"
        )
        
        self.assertIn('adjustments_applied', params)
        self.assertEqual(params['adjustments_applied']['document_type'], "technical")
        self.assertEqual(params['adjustments_applied']['performance_priority'], "accuracy")
        self.assertEqual(params['adjustments_applied']['data_size'], "large")
    
    def test_combined_adjustments(self):
        """Test that the optimizer applies multiple adjustments correctly."""
        base_params = get_enhanced_rag_parameters("Knowledge Management")
        adjusted_params = get_enhanced_rag_parameters(
            "Knowledge Management",
            document_type="technical",
            performance_priority="accuracy",
            data_size="large"
        )
        
        # Check that multiple parameters are adjusted
        differences = 0
        for key in ['chunk_size_tokens', 'embedding_model_recommendation', 'top_k', 'retrieval_enhancements']:
            if key in base_params and key in adjusted_params and base_params[key] != adjusted_params[key]:
                differences += 1
        
        self.assertGreater(
            differences,
            1,
            "Multiple parameters should be adjusted when combining document type, performance priority, and data size"
        )

if __name__ == "__main__":
    unittest.main()
