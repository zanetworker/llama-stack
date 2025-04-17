import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import base64
import io
import os
import logging
import hashlib
import json
import gc
import inspect

from mcp.server.fastmcp import FastMCP
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData, INTERNAL_ERROR, INVALID_PARAMS

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfBackend,
    PdfPipelineOptions, 
    OcrEngine, 
    EasyOcrOptions
)
from docling.datamodel.settings import settings
from docling.utils.accelerator_utils import AcceleratorDevice
from docling.utils import accelerator_utils

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure accelerator settings
def configure_accelerator():
    """Configure the accelerator device for Docling."""
    try:
        # Check if the accelerator_device attribute exists
        if hasattr(settings.perf, 'accelerator_device'):
            # Try to use MPS (Metal Performance Shaders) on macOS
            settings.perf.accelerator_device = AcceleratorDevice.MPS
            logger.info(f"Configured accelerator device: {settings.perf.accelerator_device}")
        else:
            logger.info("Accelerator device configuration not supported in this version of Docling")
        
        # Optimize batch processing
        settings.perf.doc_batch_size = 1  # Process one document at a time
        logger.info(f"Configured batch size: {settings.perf.doc_batch_size}")
        
        return True
    except Exception as e:
        logger.warning(f"Failed to configure accelerator: {e}")
        return False

# Call this function before creating the FastMCP instance
configure_accelerator()
mcp = FastMCP("docling")

# Create a cache directory
CACHE_DIR = Path.home() / ".cache" / "mcp-docling"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_cache_key(source: str, enable_ocr: bool, ocr_language: Optional[List[str]]) -> str:
    """Generate a cache key for the document conversion."""
    key_data = {
        "source": source,
        "enable_ocr": enable_ocr,
        "ocr_language": ocr_language or []
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()

def cleanup_memory():
    """Force garbage collection to free up memory."""
    gc.collect()
    logger.debug("Performed memory cleanup")

@mcp.tool()
def convert_document(
    source: str, 
    enable_ocr: bool = False,
    ocr_language: Optional[List[str]] = None
) -> str:
    """
    Convert a document from a URL or local path to markdown format.
    
    Args:
        source: URL or local file path to the document
        enable_ocr: Whether to enable OCR for scanned documents
        ocr_language: List of language codes for OCR (e.g. ["en", "fr"])
        
    Returns:
        The document content in markdown format
    
    Usage:
        convert_document("https://arxiv.org/pdf/2408.09869")
        convert_document("/path/to/document.pdf", enable_ocr=True, ocr_language=["en"])
    """
    try:
        # Remove any quotes from the source string
        source = source.strip('"\'')
        
        # Log the cleaned source
        logger.info(f"Processing document from source: {source}")
        
        # Generate cache key
        cache_key = get_cache_key(source, enable_ocr, ocr_language)
        cache_file = CACHE_DIR / f"{cache_key}.md"
        
        # Check if result is cached
        if cache_file.exists():
            logger.info(f"Using cached result for {source}")
            return cache_file.read_text()
            
        # Log the start of processing
        logger.info(f"Starting conversion of document: {source}")
        
        # Configure OCR if enabled
        format_options = {}
        if enable_ocr:
            ocr_options = EasyOcrOptions(lang=ocr_language or ["en"])
            pipeline_options = PdfPipelineOptions(
                do_ocr=True, 
                ocr_options=ocr_options,
                accelerator_device=AcceleratorDevice.MPS  # Explicitly set MPS
            )
            format_options = {
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        
        # Create converter with MPS acceleration
        logger.debug(f"Creating DocumentConverter with format_options: {format_options}")
        converter = DocumentConverter(format_options=format_options)
        
        # Convert the document
        result = converter.convert(source)
        
        # Check for errors - handle different API versions
        has_error = False
        error_message = ""
        
        # Try different ways to check for errors based on the API version
        if hasattr(result, 'status'):
            if hasattr(result.status, 'is_error'):
                has_error = result.status.is_error
            elif hasattr(result.status, 'error'):
                has_error = result.status.error
            
        if hasattr(result, 'errors') and result.errors:
            has_error = True
            error_message = str(result.errors)
        
        if has_error:
            error_msg = f"Conversion failed: {error_message}"
            raise McpError(ErrorData(INTERNAL_ERROR, error_msg))
            
        # Export to markdown
        markdown_output = result.document.export_to_markdown()
        
        # Cache the result
        cache_file.write_text(markdown_output)
        
        # Log completion
        logger.info(f"Successfully converted document: {source}")
        
        # Clean up memory
        cleanup_memory()
        
        return markdown_output
        
    except Exception as e:
        logger.exception(f"Error converting document: {source}")
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}"))

@mcp.tool()
def convert_document_with_images(
    source: str, 
    enable_ocr: bool = False,
    ocr_language: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convert a document from a URL or local path to markdown format and return embedded images.
    
    Args:
        source: URL or local file path to the document
        enable_ocr: Whether to enable OCR for scanned documents
        ocr_language: List of language codes for OCR (e.g. ["en", "fr"])
        
    Returns:
        A dictionary containing the markdown text and a list of base64-encoded images
        
    Usage:
        convert_document_with_images("https://arxiv.org/pdf/2408.09869")
    """
    try:
        # Remove any quotes from the source string
        source = source.strip('"\'')
        
        # Configure OCR if enabled
        format_options = {}
        if enable_ocr:
            ocr_options = EasyOcrOptions(lang=ocr_language or ["en"])
            pipeline_options = PdfPipelineOptions(do_ocr=True, ocr_options=ocr_options)
            format_options = {
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        
        # Create converter and convert document
        converter = DocumentConverter(format_options=format_options)
        result = converter.convert(source)
        
        # Check for errors - handle different API versions
        has_error = False
        error_message = ""
        
        # Try different ways to check for errors based on the API version
        if hasattr(result, 'status'):
            if hasattr(result.status, 'is_error'):
                has_error = result.status.is_error
            elif hasattr(result.status, 'error'):
                has_error = result.status.error
            
        if hasattr(result, 'errors') and result.errors:
            has_error = True
            error_message = str(result.errors)
        
        if has_error:
            error_msg = f"Conversion failed: {error_message}"
            raise McpError(ErrorData(INTERNAL_ERROR, error_msg))
            
        # Export to markdown
        markdown_output = result.document.export_to_markdown()
        
        # Extract images
        images = []
        for item in result.document.items:
            if hasattr(item, 'get_image') and callable(getattr(item, 'get_image')):
                try:
                    img = item.get_image(result.document)
                    if img:
                        buffered = io.BytesIO()
                        img.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        images.append({
                            "id": item.id,
                            "data": img_str,
                            "format": "png"
                        })
                except Exception:
                    # Skip images that can't be processed
                    pass
        
        return {
            "markdown": markdown_output,
            "images": images
        }
        
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}"))

@mcp.tool()
def extract_tables(
    source: str
) -> List[str]:
    """
    Extract tables from a document and return them as structured data.
    
    Args:
        source: URL or local file path to the document
        
    Returns:
        A list of tables in markdown format
        
    Usage:
        extract_tables("https://arxiv.org/pdf/2408.09869")
    """
    source = source.strip('"\'')

    # Create converter and convert document
    converter = DocumentConverter()

    # The issue might be in how the source is passed to convert
    # Let's ensure it's passed as a keyword argument
    conversion_result = converter.convert(source=source)
    tables_results = []
    for table in conversion_result.document.tables:
        tables_results.append(table.export_to_markdown())

    return tables_results


@mcp.tool()
def convert_batch(
    sources: List[str],
    enable_ocr: bool = False,
    ocr_language: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Convert multiple documents in batch mode.
    
    Args:
        sources: List of URLs or file paths to documents
        enable_ocr: Whether to enable OCR for scanned documents
        ocr_language: List of language codes for OCR (e.g. ["en", "fr"])
        
    Returns:
        Dictionary mapping source paths to their markdown content
        
    Usage:
        convert_batch(["https://arxiv.org/pdf/2408.09869", "/path/to/document.pdf"])
    """
    try:
        # Configure OCR if enabled
        format_options = {}
        if enable_ocr:
            ocr_options = EasyOcrOptions(lang=ocr_language or ["en"])
            pipeline_options = PdfPipelineOptions(do_ocr=True, ocr_options=ocr_options)
            format_options = {
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        
        # Create converter
        converter = DocumentConverter(format_options=format_options)
        
        # Process each document
        results = {}
        for source in sources:
            # Remove any quotes from the source string
            source = source.strip('"\'')
            logger.info(f"Processing document from source: {source}")
            
            try:
                result = converter.convert(source)
                
                # Check for errors - handle different API versions
                has_error = False
                error_message = ""
                
                # Try different ways to check for errors based on the API version
                if hasattr(result, 'status'):
                    if hasattr(result.status, 'is_error'):
                        has_error = result.status.is_error
                    elif hasattr(result.status, 'error'):
                        has_error = result.status.error
                    
                if hasattr(result, 'errors') and result.errors:
                    has_error = True
                    error_message = str(result.errors)
                
                if has_error:
                    results[source] = f"Error: {error_message}"
                else:
                    results[source] = result.document.export_to_markdown()
            except Exception as e:
                results[source] = f"Error: {str(e)}"
        
        return results
        
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}"))

@mcp.tool()
def get_system_info() -> Dict[str, Any]:
    """
    Get information about the system configuration and acceleration status.
    
    Returns:
        Dictionary with system information
    
    Usage:
        get_system_info()
    """
    try:
        system_info = {
            "batch_settings": {
                "doc_batch_size": settings.perf.doc_batch_size,
                "doc_batch_concurrency": settings.perf.doc_batch_concurrency
            },
            "cache": {
                "enabled": True,
                "location": str(CACHE_DIR)
            }
        }
        
        # Add accelerator info if available
        if hasattr(settings.perf, 'accelerator_device'):
            system_info["accelerator"] = {
                "configured": str(settings.perf.accelerator_device),
                "available": ["CPU", "MPS"]  # Hardcode the common options
            }
        else:
            system_info["accelerator"] = {
                "configured": "Not configured",
                "available": ["CPU"]  # Default to CPU only
            }
            
        return system_info
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Error getting system info: {str(e)}")) 