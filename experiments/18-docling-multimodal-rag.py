#!/usr/bin/env python3
"""
Multimodal RAG with Docling and Llama Stack

This script demonstrates how to build a multimodal RAG system using Docling for document 
processing and Llama Stack for embeddings, vector storage, and LLM capabilities.

It processes PDF documents (including text, tables, and images) and enables querying the content.
The script separates ingestion and retrieval processes for better modularity.
"""

import os
import asyncio
import tempfile
import base64
import io
import argparse
from pathlib import Path
from PIL import Image, ImageOps
from termcolor import cprint

# Llama Stack imports
from llama_stack_client import LlamaStackClient
from llama_stack_client.types import Document

# Docling imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.types.doc.document import TableItem
from docling_core.types.doc.labels import DocItemLabel


def encode_image(image: Image.Image, format: str = "png") -> str:
    """Convert an image to base64 for API consumption"""
    image = ImageOps.exif_transpose(image).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format)
    return f"data:image/{format};base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"


def get_llama_stack_client():
    """Create and return a Llama Stack client instance"""
    llama_stack_port = os.environ.get("LLAMA_STACK_PORT", "8080")
    return LlamaStackClient(base_url=f"http://localhost:{llama_stack_port}")


async def ingest_document(client, doc_url, vector_db_id, embedding_model, embedding_dimension, vision_model_id=None):
    """
    Process a document with Docling and ingest it into a vector database
    
    Args:
        client: LlamaStackClient instance
        doc_url: URL or path to the document
        vector_db_id: ID for the vector database
        embedding_model: Model to use for embeddings
        embedding_dimension: Dimension of the embeddings
        vision_model_id: Optional model to use for image analysis
        
    Returns:
        Dictionary with statistics about the ingestion process
    """
    # Initialize Docling document converter
    cprint(f"Processing document from: {doc_url}", "cyan")
    pdf_pipeline_options = PdfPipelineOptions(do_ocr=False, generate_picture_images=True)
    format_options = {InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options)}
    converter = DocumentConverter(format_options=format_options)
    
    # Convert document
    cprint("Converting document with Docling...", "cyan")
    conversion_result = converter.convert(source=doc_url)
    docling_document = conversion_result.document
    cprint(f"Document converted successfully, found {len(docling_document.paragraphs)} paragraphs, " 
           f"{len(docling_document.tables)} tables, {len(docling_document.pictures)} images",
           "green")
    
    # Create vector database if it doesn't exist
    try:
        # Check if vector db exists
        client.vector_dbs.get(vector_db_id=vector_db_id)
        cprint(f"Using existing vector database: {vector_db_id}", "yellow")
    except Exception:
        # Register new vector database
        cprint(f"Registering vector database with embedding model: {embedding_model}", "cyan")
        client.vector_dbs.register(
            vector_db_id=vector_db_id,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            provider_id="faiss"
        )
    
    # Process and store text chunks
    cprint("Processing text chunks...", "cyan")
    text_documents = []
    doc_id = 0
    
    # For this example, we'll just use chunk size in characters
    chunk_size = 1024
    
    for chunk in HybridChunker(chunk_size=chunk_size).chunk(docling_document):
        items = chunk.meta.doc_items
        if len(items) == 1 and isinstance(items[0], TableItem):
            continue  # Tables will be processed separately
            
        refs = " ".join(map(lambda item: item.get_ref().cref, items))
        document = Document(
            document_id=f"text-{doc_id}",
            content=chunk.text,
            mime_type="text/plain",
            metadata={
                "source": doc_url, 
                "ref": refs, 
                "type": "text",
                "doc_type": "pdf"
            }
        )
        text_documents.append(document)
        doc_id += 1
    
    cprint(f"Created {len(text_documents)} text documents", "green")
    
    # Process tables
    cprint("Processing tables...", "cyan")
    table_documents = []
    doc_id = 0
    
    for table in docling_document.tables:
        if table.label in [DocItemLabel.TABLE]:
            ref = table.get_ref().cref
            text = table.export_to_markdown()
            document = Document(
                document_id=f"table-{doc_id}",
                content=text,
                mime_type="text/markdown",
                metadata={
                    "source": doc_url, 
                    "ref": ref, 
                    "type": "table",
                    "doc_type": "pdf"
                }
            )
            table_documents.append(document)
            doc_id += 1
    
    cprint(f"Created {len(table_documents)} table documents", "green")
    
    # Process images
    cprint("Processing images...", "cyan")
    image_documents = []
    doc_id = 0
    
    image_prompt = "If the image contains text, explain the text in the image."
    
    for picture in docling_document.pictures:
        ref = picture.get_ref().cref
        image = picture.get_image(docling_document)
        
        if image:
            # Get image description using the vision model or use placeholder
            try:
                if vision_model_id:
                    # In a real implementation, you would call the vision model here
                    # This is a placeholder for the actual implementation
                    text = f"Image description for {ref}. In a real application, this would be generated by the vision model."
                else:
                    text = f"Image at {ref} (description not available without vision model)"
                
                document = Document(
                    document_id=f"image-{doc_id}",
                    content=text,
                    mime_type="text/plain",
                    metadata={
                        "source": doc_url, 
                        "ref": ref, 
                        "type": "image",
                        "doc_type": "pdf"
                    }
                )
                image_documents.append(document)
                doc_id += 1
            except Exception as e:
                cprint(f"Error processing image {ref}: {str(e)}", "red")
    
    cprint(f"Created {len(image_documents)} image descriptions", "green")
    
    # Combine all documents
    all_documents = text_documents + table_documents + image_documents
    cprint(f"Total documents: {len(all_documents)}", "green")
    
    # Insert documents into vector database
    cprint("Inserting documents into vector database...", "cyan")
    insert_response = client.tool_runtime.rag_tool.insert(
        documents=all_documents,
        vector_db_id=vector_db_id,
        chunk_size_in_tokens=256,  # Adjust as needed
    )
    cprint("Documents inserted successfully", "green")
    
    # Return statistics about the ingestion
    return {
        "doc_url": doc_url,
        "text_chunks": len(text_documents),
        "table_chunks": len(table_documents),
        "image_chunks": len(image_documents),
        "total_chunks": len(all_documents),
        "vector_db_id": vector_db_id
    }


async def retrieve_and_generate(client, vector_db_id, query_text, llm_model_id):
    """
    Retrieve relevant chunks for a query and generate a response
    
    Args:
        client: LlamaStackClient instance
        vector_db_id: ID of the vector database to query
        query_text: The user query
        llm_model_id: ID of the LLM to use for response generation
        
    Returns:
        Dictionary with the query results
    """
    cprint(f"Querying: {query_text}", "cyan")
    
    # Retrieve relevant chunks
    response = client.tool_runtime.rag_tool.query(
        content=[{
            "type": "text",
            "text": query_text
        }],
        vector_db_ids=[vector_db_id],
        max_chunks=5
    )
    
    # Format retrieved chunks for display
    if hasattr(response, 'chunks'):
        cprint("Retrieved chunks:", "green")
        chunks_info = []
        
        # Format context from chunks for LLM
        context = ""
        
        for i, chunk in enumerate(response.chunks):
            chunk_info = {
                "index": i + 1,
                "score": chunk.score,
                "type": chunk.metadata.get('type', 'unknown'),
                "source": chunk.metadata.get('source', 'unknown'),
                "content_preview": chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content
            }
            chunks_info.append(chunk_info)
            
            # Print chunk info
            cprint(f"Chunk {i+1}:", "cyan")
            cprint(f"  Score: {chunk.score}", "yellow")
            cprint(f"  Type: {chunk.metadata.get('type', 'unknown')}", "yellow")
            cprint(f"  Content: {chunk_info['content_preview']}", "white")
            
            # Add to context for LLM
            context += f"\n\nChunk {i+1} ({chunk.metadata.get('type', 'text')}):\n{chunk.content}"
    
        # Generate response using LLM if model_id is provided
        if llm_model_id:
            cprint("\nGenerating response with LLM...", "cyan")
            try:
                # Create system prompt
                system_prompt = """You are a helpful assistant answering questions about documents.
                Use only the provided context chunks to answer the question.
                If the context doesn't contain relevant information, say so clearly.
                For tables, explain them in a clear, structured way.
                For images, refer to their descriptions as provided in the context."""
                
                # Call LLM API
                llm_response = client.chat.completions.create(
                    model=llm_model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query_text}"}
                    ]
                )
                
                # Extract and return generated answer
                answer = llm_response.choices[0].message.content
                cprint("\nLLM Response:", "green")
                cprint(answer, "white")
                
                return {
                    "query": query_text,
                    "chunks": chunks_info,
                    "answer": answer
                }
            except Exception as e:
                cprint(f"Error generating LLM response: {str(e)}", "red")
                # Continue without LLM response
        
        # If no LLM is used or there was an error, return just the chunks
        return {
            "query": query_text,
            "chunks": chunks_info,
            "answer": None
        }
    else:
        cprint("No chunks retrieved", "yellow")
        return {
            "query": query_text,
            "chunks": [],
            "answer": None
        }


async def main():
    """Main function demonstrating separate ingestion and retrieval"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Multimodal RAG with Docling and Llama Stack")
    parser.add_argument("--mode", choices=["ingest", "retrieve", "demo"], default="demo",
                      help="Mode of operation: ingest, retrieve, or demo (runs both)")
    parser.add_argument("--doc-url", default="https://midwestfoodbank.org/images/AR_2020_WEB2.pdf",
                      help="URL or path to the document for ingestion")
    parser.add_argument("--query", default="How much was spent on food distribution?",
                      help="Query for retrieval mode")
    parser.add_argument("--vector-db-id", default="docling-multimodal-rag",
                      help="ID for the vector database")
    parser.add_argument("--clean", action="store_true",
                      help="Clean existing vector database before ingestion")
    args = parser.parse_args()
    
    # Initialize client
    client = get_llama_stack_client()
    
    # Get model IDs from environment or use defaults
    llm_model_id = os.environ.get("INFERENCE_MODEL", "Meta-Llama-3-8B-Instruct")
    vision_model_id = os.environ.get("VISION_MODEL", "granite-vision-3.2-2b") 
    embedding_model = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    embedding_dimension = int(os.environ.get("EMBEDDING_DIMENSION", "384"))
    
    cprint(f"Using LLM model: {llm_model_id}", "green")
    cprint(f"Using vision model: {vision_model_id}", "green")
    cprint(f"Using embedding model: {embedding_model}", "green")
    
    # Clean existing vector db if requested
    if args.clean and (args.mode == "ingest" or args.mode == "demo"):
        try:
            client.vector_dbs.delete(vector_db_id=args.vector_db_id)
            cprint(f"Deleted existing vector database: {args.vector_db_id}", "yellow")
        except Exception:
            pass
    
    # Perform operations based on selected mode
    if args.mode == "ingest" or args.mode == "demo":
        # Ingestion process
        cprint("\n===== DOCUMENT INGESTION =====", "magenta")
        stats = await ingest_document(
            client, 
            args.doc_url, 
            args.vector_db_id, 
            embedding_model, 
            embedding_dimension,
            vision_model_id
        )
        cprint("\nIngestion Statistics:", "cyan")
        for key, value in stats.items():
            cprint(f"  {key}: {value}", "white")
    
    if args.mode == "retrieve" or args.mode == "demo":
        # Retrieval process
        cprint("\n===== QUERY RETRIEVAL =====", "magenta")
        queries = [args.query]
        
        # Add more example queries if in demo mode
        if args.mode == "demo":
            queries.extend([
                "What tables are in the document?",
                "Describe the main images in the document"
            ])
        
        for query in queries:
            cprint(f"\nProcessing query: {query}", "cyan")
            result = await retrieve_and_generate(client, args.vector_db_id, query, llm_model_id)
            
            # If no LLM response was generated, show how it would be done
            if not result["answer"]:
                cprint("\nTo generate complete answers with the LLM, you would:", "green")
                cprint("1. Format the retrieved chunks as context", "white")
                cprint("2. Send the query + context to the LLM", "white")
                cprint("3. Return the generated response", "white")
                cprint("\nExample with LlamaStackClient:", "green")
                cprint("response = client.chat.completions.create(", "yellow")
                cprint("    model=llm_model_id,", "yellow")
                cprint("    messages=[", "yellow")
                cprint("        {\"role\": \"system\", \"content\": \"Answer based on the context provided.\"},", "yellow")
                cprint("        {\"role\": \"user\", \"content\": f\"Context: {context}\\n\\nQuestion: {query}\"}]", "yellow")
                cprint(")\n", "yellow")


if __name__ == "__main__":
    asyncio.run(main()) 