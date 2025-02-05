import argparse
import logging
from pathlib import Path
from typing import List, Tuple
import yaml
import os
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import PictureItem, TableItem
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from termcolor import cprint  # Add colored output for better visibility
import datetime

# LlamaStack imports
from llama_stack_client import LlamaStackClient
from llama_stack_client.types import Document
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_rag_connection(vector_db_id: str, chunk_size: int = 256):
    """Initialize LlamaStack connection and vector DB"""
    try:
        client = LlamaStackClient(
            base_url=f"http://localhost:{os.environ.get('LLAMA_STACK_PORT', '8080')}"
        )
        cprint(f"Connected to LlamaStack at port {os.environ.get('LLAMA_STACK_PORT', '8080')}", "green")
        
        # Clean up existing vector db
        try:
            client.vector_dbs.delete(vector_db_id=vector_db_id)
            cprint(f"Cleaned up existing vector DB: {vector_db_id}", "yellow")
        except Exception as e:
            cprint(f"Note: Vector DB cleanup skipped: {e}", "yellow")

        # Register new vector db
        client.vector_dbs.register(
            vector_db_id=vector_db_id,
            embedding_model="all-MiniLM-L6-v2",
            embedding_dimension=384,
        )
        cprint(f"Registered new vector DB: {vector_db_id}", "green")
        
        return client
    except Exception as e:
        cprint(f"Failed to setup RAG connection: {e}", "red")
        raise

def ingest_documents_into_rag(client, documents: List[Document], vector_db_id: str):
    """Insert processed documents into RAG system"""
    try:
        cprint(f"Starting ingestion of {len(documents)} documents into vector DB", "blue")
        client.tool_runtime.rag_tool.insert(
            documents=documents,
            vector_db_id=vector_db_id,
            chunk_size_in_tokens=256,
        )
        cprint(f"Successfully ingested {len(documents)} documents into RAG system", "green")
    except Exception as e:
        cprint(f"Error ingesting documents: {e}", "red")
        raise

def process_documents_to_rag(input_dir: Path) -> List[Document]:
    """Process documents and convert to RAG-ready format"""
    cprint(f"Processing documents from: {input_dir}", "blue")
    
    # Get all document files recursively
    documents_to_process, markdown_files = get_document_files(input_dir)
    cprint(f"Found {len(documents_to_process)} documents to process", "blue")
    
    # Configure pipeline
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_page_images = False
    pipeline_options.generate_table_images = False
    pipeline_options.generate_picture_images = True

    doc_converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.PPTX],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,
                backend=PyPdfiumDocumentBackend,
                pipeline_options=pipeline_options,
            ),
            InputFormat.DOCX: WordFormatOption(pipeline_cls=SimplePipeline),
        },
    )

    rag_documents = []
    
    try:
        # Process and convert documents - convert generator to list
        conv_results = list(doc_converter.convert_all(documents_to_process))
        cprint(f"Successfully converted {len(conv_results)} documents", "green")

        # Create RAG documents with metadata
        for i, res in enumerate(conv_results):
            content = res.document.export_to_markdown()
            cprint(f"Processing document {i+1}/{len(conv_results)}: {res.input.file.name}", "blue")
            
            rag_doc = Document(
                document_id=f"doc_{i}",
                content=content,
                mime_type="text/markdown",
                metadata={
                    "source_path": str(res.input.file),
                    "processed_at": datetime.datetime.now().isoformat()
                }
            )
            rag_documents.append(rag_doc)
            cprint(f"Created RAG document {i+1} with {len(content)} characters", "green")
        
        return rag_documents
    except Exception as e:
        cprint(f"Error processing documents: {e}", "red")
        raise

def get_document_files(input_dir: Path) -> Tuple[List[Path], List[Path]]:
    """
    Recursively scan directory for document files.
    Returns:
        tuple: (documents_to_process, markdown_files)
    """
    process_extensions = {".pdf", ".docx", ".pptx"}
    documents_to_process = []
    markdown_files = []

    for path in input_dir.rglob("*"):
        if path.is_file():
            if path.suffix.lower() in process_extensions:
                documents_to_process.append(path)
            elif path.suffix.lower() == ".md":
                markdown_files.append(path)

    return documents_to_process, markdown_files

def setup_rag_agent(client, vector_db_id: str):
    """Configure and create RAG-enabled agent"""
    try:
        # Configure agent with context window settings
        agent_config = AgentConfig(
            model=os.environ.get('INFERENCE_MODEL', 'llama2'),
            instructions="""You are a knowledgeable assistant. Your task is to:
                        1. Use ONLY the retrieved context to answer queries
                        2. Provide accurate, focused responses
                        3. If you can't find relevant information in the context, say so
                        4. Do not make up or infer information not present in the context""",
            enable_session_persistence=False,
            max_infer_iters=2,
            toolgroups=[
                {
                    "name": "builtin::rag",
                    "args": {
                        "vector_db_ids": [vector_db_id],
                        "query_config": {
                            "max_tokens_in_context": 1800,
                            "max_chunks": 3,
                            "query_generator_config": {
                                "type": "default",
                                "separator": "\n"
                            }
                        }
                    }
                }
            ],
        )

        # Create agent and session
        rag_agent = Agent(client, agent_config)
        session_id = rag_agent.create_session("test-session")
        cprint(f"Created RAG agent with session ID: {session_id}", "green")
        
        return rag_agent, session_id
    except Exception as e:
        cprint(f"Error setting up RAG agent: {e}", "red")
        raise

def chat_with_documents(rag_agent, session_id: str):
    """Interactive chat session with RAG-enabled agent"""
    cprint("\nStarting chat session (type 'exit' to end)", "blue")
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou> ")
            if user_input.lower() in ['exit', 'quit']:
                break

            cprint("\nAssistant> ", "green", end='')
            
            # Create agent turn
            response = rag_agent.create_turn(
                messages=[{
                    "role": "user",
                    "content": user_input
                }],
                session_id=session_id
            )
            
            # Handle streaming response
            for log in EventLogger().log(response):
                try:
                    log.print()
                except Exception as e:
                    cprint(f"\nError printing response: {str(e)}", "red")
                    continue

        except Exception as e:
            cprint(f"\nError in chat: {str(e)}", "red")
            continue

def main():
    parser = argparse.ArgumentParser(description="Process documents and chat with RAG")
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory containing documents")
    parser.add_argument("--vector_db_id", type=str, default="document_rag", help="Vector DB ID for document storage")
    parser.add_argument("--chunk_size", type=int, default=256, help="Chunk size for document processing")
    args = parser.parse_args()

    try:
        # Process documents
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
            
        cprint(f"Starting document processing from: {input_dir}", "blue")
        rag_documents = process_documents_to_rag(input_dir)
        
        # Set up RAG connection
        client = setup_rag_connection(args.vector_db_id, args.chunk_size)
        
        # Ingest into RAG system
        ingest_documents_into_rag(client, rag_documents, args.vector_db_id)
        
        cprint(f"Successfully processed and ingested {len(rag_documents)} documents", "green")
        
        # Set up RAG agent
        rag_agent, session_id = setup_rag_agent(client, args.vector_db_id)
        
        # Start interactive chat
        chat_with_documents(rag_agent, session_id)
        
    except Exception as e:
        cprint(f"Error in main execution: {e}", "red")
        raise

if __name__ == "__main__":
    main()