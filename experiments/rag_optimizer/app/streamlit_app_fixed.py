import streamlit as st
import json
import pandas as pd
import re
import plotly.graph_objects as go
import sys
import os
import tempfile
from typing import List, Dict, Any
import html

# Add the parent directory to the path to import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import the enhanced RAG parameter optimizer
from experiments.rag_optimizer.core.enhanced_optimizer import get_rag_parameters
from experiments.rag_optimizer.integration.pipeline_example import (
    parse_chunk_size, parse_overlap, parse_top_k,
    map_embedding_recommendation, map_reranker_recommendation,
    parse_temperature, map_llm_recommendation
)

# Import Llama Stack integration
from experiments.rag_optimizer.integration.llama_stack_integration import LlamaStackRAGIntegration
from llama_stack_client.types import Document

# Function to configure RAG pipeline based on parameters
def configure_rag_pipeline(parameters):
    """
    Configure a RAG pipeline based on the provided parameters.
    
    Args:
        parameters: Dictionary of RAG parameters
        
    Returns:
        Dictionary containing the configured pipeline components
    """
    # Parse parameters
    chunk_size = parse_chunk_size(parameters.get('chunk_size_tokens', '~1000'))
    overlap = parse_overlap(parameters.get('overlap_tokens', '~100'))
    chunk_strategy = parameters.get('chunking_strategy_recommendation', 'Recursive/Semantic').split('/')[0].lower()
    embed_model_name = map_embedding_recommendation(parameters.get('embedding_model_recommendation', ''))
    top_k = parse_top_k(parameters.get('top_k', '5'))
    retrieval_enhancements = parameters.get('retrieval_enhancements', '')
    
    # Return configuration details
    return {
        "text_splitter": {
            "chunk_size": chunk_size,
            "chunk_overlap": overlap,
            "strategy": chunk_strategy
        },
        "embedding_model": {
            "model_name": embed_model_name
        },
        "retriever": {
            "top_k": top_k,
            "search_type": 'hybrid' if 'hybrid' in retrieval_enhancements.lower() else 'similarity'
        },
        "reranker": map_reranker_recommendation(retrieval_enhancements),
        "llm": {
            "model_name": map_llm_recommendation(parameters.get('generation_settings', '')),
            "temperature": parse_temperature(parameters.get('generation_settings', ''))
        }
    }

# Function to extract parameters from chat messages
def extract_parameters_from_chat(messages):
    """
    Extract RAG parameters from chat messages.
    
    Args:
        messages: List of chat messages
        
    Returns:
        Dictionary containing extracted parameters
    """
    # Default values
    use_case = None
    document_type = "general"
    performance_priority = "balanced"
    data_size = "medium"
    
    # Extract use case
    use_case_patterns = [
        r"use case(?:\s+is)?(?:\s+for)?\s*:?\s*([\w\s]+)",
        r"for\s+([\w\s]+)\s+use case",
        r"(knowledge management|customer support|healthcare|education|code assistance|sales automation|marketing|threat analysis|gaming)"
    ]
    
    for message in messages:
        if message["role"] == "user":
            content = message["content"].lower()
            
            # Extract use case
            if not use_case:
                for pattern in use_case_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        use_case = matches[0].strip().title()
                        break
            
            # Extract document type
            if "technical" in content or "scientific" in content:
                document_type = "technical"
            elif "legal" in content or "contract" in content or "regulatory" in content:
                document_type = "legal"
            elif "education" in content or "learning" in content or "teaching" in content:
                document_type = "educational"
            elif "code" in content or "programming" in content or "software" in content:
                document_type = "code"
            
            # Extract performance priority
            if "accuracy" in content or "precise" in content or "accurate" in content:
                performance_priority = "accuracy"
            elif "latency" in content or "fast" in content or "speed" in content or "quick" in content:
                performance_priority = "latency"
            elif "cost" in content or "cheap" in content or "budget" in content or "efficient" in content:
                performance_priority = "cost"
            
            # Extract data size
            if "small" in content or "few" in content:
                data_size = "small"
            elif "large" in content or "big" in content or "huge" in content:
                data_size = "large"
    
    return {
        "use_case": use_case,
        "document_type": document_type,
        "performance_priority": performance_priority,
        "data_size": data_size
    }

# Function to generate Mermaid diagram for RAG pipeline
def generate_mermaid_diagram(pipeline_config):
    """
    Generate a Mermaid diagram for the RAG pipeline.
    
    Args:
        pipeline_config: Dictionary containing pipeline configuration
        
    Returns:
        String containing Mermaid diagram code
    """
    # Create Mermaid diagram
    mermaid_code = """
    graph LR
        A[Documents] --> B[Text Splitter]
        B --> C[Chunks]
        C --> D[Embedding Model]
        D --> E[Vector Store]
        E --> F[Retriever]
        F --> G[Retrieved Chunks]
    """
    
    # Add reranker if available
    if pipeline_config["reranker"]:
        mermaid_code += """
        G --> H[Reranker]
        H --> I[Ranked Chunks]
        I --> J[LLM]
        """
    else:
        mermaid_code += """
        G --> J[LLM]
        """
    
    mermaid_code += """
        K[Query] --> F
        K --> J
        J --> L[Response]
        
        classDef config fill:#f9f,stroke:#333,stroke-width:2px;
        class B,D,F,H,J config;
    """
    
    return mermaid_code

# Function to process uploaded files
def process_uploaded_files(uploaded_files):
    """
    Process uploaded files and convert them to Llama Stack Document objects.
    
    Args:
        uploaded_files: List of uploaded files from Streamlit
        
    Returns:
        List of Document objects
    """
    documents = []
    
    for i, file in enumerate(uploaded_files):
        # Read file content
        content = file.read()
        
        # Convert to string if binary
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        
        # Create Document object
        doc = Document(
            document_id=f"doc_{i}",
            content=content,
            mime_type=file.type if file.type else "text/plain",
            metadata={"filename": file.name}
        )
        
        documents.append(doc)
    
    return documents

# Function to safely display text that might contain problematic Markdown
def safe_display(text):
    """
    Safely display text that might contain problematic Markdown directives.
    
    Args:
        text: The text to display
        
    Returns:
        None (displays the text using Streamlit)
    """
    try:
        # Try to display as Markdown first
        st.markdown(text)
    except Exception as e:
        # If Markdown rendering fails, display as plain text
        st.text(text)
        st.warning(f"Note: Markdown rendering failed, displaying as plain text. Error: {str(e)}")

def main():
    """Main function for the Streamlit application."""
    st.set_page_config(layout="wide", page_title="RAG Parameter Optimizer")
    
    st.title("RAG Parameter Optimizer")
    st.markdown("""
    This application helps you optimize your RAG (Retrieval-Augmented Generation) parameters 
    based on your specific use case and requirements. Choose between the interfaces below to get started.
    """)
    
    # Create tabs for different interfaces
    tab1, tab2, tab3 = st.tabs(["Form Interface", "Chat Interface", "Llama Stack Testing"])
    
    # Form Interface
    with tab1:
        st.header("Form-based Parameter Optimization")
        st.write("Select your requirements using the form below to get optimized RAG parameters.")
        
        # Sidebar for inputs
        with st.sidebar:
            st.header("Configuration")
            
            # Use case selection
            use_cases = [
                "Knowledge Management", 
                "Customer Support", 
                "Healthcare Applications", 
                "Education", 
                "Code Assistance", 
                "Sales Automation", 
                "Marketing", 
                "Threat Analysis", 
                "Gaming"
            ]
            use_case = st.selectbox("Select Use Case", use_cases, key="form_use_case")
            
            # Document type selection
            document_types = ["general", "technical", "legal", "educational", "code"]
            document_type = st.selectbox("Document Type", document_types, key="form_doc_type")
            
            # Performance priority selection
            performance_priorities = ["balanced", "accuracy", "latency", "cost"]
            performance_priority = st.selectbox("Performance Priority", performance_priorities, key="form_perf_priority")
            
            # Data size selection
            data_sizes = ["small", "medium", "large"]
            data_size = st.selectbox("Data Size", data_sizes, key="form_data_size")
            
            # Button to optimize parameters
            optimize_button = st.button("Optimize Parameters")
        
        # Main content area
        if optimize_button:
            # Get optimized parameters
            parameters = get_rag_parameters(use_case, document_type, performance_priority, data_size)
            
            # Display parameters
            st.header("Optimized RAG Parameters")
            
            # Display parameters as JSON
            st.json(parameters)
            
            # Configure pipeline
            pipeline_config = configure_rag_pipeline(parameters)
            
            # Display pipeline configuration
            st.header("RAG Pipeline Configuration")
            
            # Create columns for different components
            col1, col2 = st.columns(2)
            
            with col1:
                # Display text splitter configuration
                st.subheader("Text Splitter")
                st.json(pipeline_config["text_splitter"])
                
                # Display embedding model configuration
                st.subheader("Embedding Model")
                st.json(pipeline_config["embedding_model"])
            
            with col2:
                # Display retriever configuration
                st.subheader("Retriever")
                st.json(pipeline_config["retriever"])
                
                # Display reranker configuration if available
                if pipeline_config["reranker"]:
                    st.subheader("Reranker")
                    st.write(f"Using reranker: {pipeline_config['reranker']}")
                
                # Display LLM configuration
                st.subheader("LLM")
                st.json(pipeline_config["llm"])
            
            # Display pipeline visualization
            st.header("RAG Pipeline Visualization")
            
            # Generate Mermaid diagram
            mermaid_code = generate_mermaid_diagram(pipeline_config)
            
            # Display Mermaid diagram
            try:
                # Try to import streamlit-mermaid
                from streamlit_mermaid import st_mermaid
                st_mermaid(mermaid_code)
            except ImportError:
                st.code(mermaid_code, language="mermaid")
                st.warning("Install streamlit-mermaid for interactive diagrams: `pip install streamlit-mermaid`")
            
            # Display parameter explanations
            st.header("Parameter Explanations")
            
            explanations = {
                "chunk_size_tokens": "The size of each text chunk in tokens. Larger chunks provide more context but may reduce retrieval precision.",
                "overlap_tokens": "The number of overlapping tokens between adjacent chunks. Higher overlap helps maintain context across chunk boundaries.",
                "embedding_model_recommendation": "The recommended embedding model for converting text to vector representations.",
                "top_k": "The number of chunks to retrieve for each query. Higher values provide more context but may introduce noise.",
                "retrieval_enhancements": "Additional techniques to improve retrieval quality beyond basic vector search.",
                "generation_settings": "Settings for the language model that generates the final response.",
                "prompting_technique": "Recommended prompting strategy for the language model.",
                "output_guardrails": "Safeguards to ensure the quality and safety of generated responses.",
                "chunking_strategy_recommendation": "The recommended approach for dividing documents into chunks."
            }
            
            # Create columns for explanations
            col1, col2 = st.columns(2)
            
            # Split explanations between columns
            params_list = list(parameters.items())
            half_point = len(params_list) // 2
            
            for i, (param, value) in enumerate(params_list):
                if param in explanations:
                    with col1 if i < half_point else col2:
                        st.subheader(param)
                        st.write(f"**Value:** {value}")
                        st.write(f"**Explanation:** {explanations[param]}")
                        
                        # Add adjustment explanations based on document type, performance priority, and data size
                        if param == "chunk_size_tokens" and document_type != "general":
                            st.write(f"**Adjustment for {document_type} documents:** Chunk size was adjusted to optimize for {document_type} content.")
                        
                        if param == "embedding_model_recommendation" and performance_priority != "balanced":
                            st.write(f"**Adjustment for {performance_priority} priority:** Embedding model was selected to prioritize {performance_priority}.")
                        
                        if param == "top_k" and data_size != "medium":
                            st.write(f"**Adjustment for {data_size} data size:** Top-k value was adjusted to account for {data_size} data corpus.")
            
            # Allow parameter customization
            st.header("Customize Parameters")
            st.write("Adjust parameters to fine-tune your RAG pipeline.")
            
            # Create columns for customization
            col1, col2 = st.columns(2)
            
            with col1:
                # Example of parameter customization
                custom_chunk_size = st.slider("Chunk Size (tokens)", 
                                            min_value=256, 
                                            max_value=4096, 
                                            value=parse_chunk_size(parameters.get('chunk_size_tokens', '~1000')),
                                            step=128)
                
                custom_overlap = st.slider("Chunk Overlap (tokens)", 
                                        min_value=0, 
                                        max_value=512, 
                                        value=parse_overlap(parameters.get('overlap_tokens', '~100')),
                                        step=32)
            
            with col2:
                custom_top_k = st.slider("Top-K", 
                                        min_value=1, 
                                        max_value=20, 
                                        value=parse_top_k(parameters.get('top_k', '5')))
                
                # Add more customization options as needed
            
            # Display impact of customization
            if st.button("Apply Custom Parameters"):
                st.subheader("Impact of Customization")
                
                # Create a radar chart to visualize the impact
                categories = ['Accuracy', 'Latency', 'Cost', 'Context Coverage', 'Relevance']
                
                # Calculate impact scores (simplified example)
                original_scores = [0.7, 0.6, 0.8, 0.7, 0.7]  # Example scores
                
                # Adjust scores based on customization
                custom_scores = original_scores.copy()
                
                # Chunk size impact
                original_chunk_size = parse_chunk_size(parameters.get('chunk_size_tokens', '~1000'))
                if custom_chunk_size > original_chunk_size:
                    custom_scores[0] += 0.1  # Accuracy
                    custom_scores[1] -= 0.1  # Latency
                    custom_scores[2] -= 0.1  # Cost
                    custom_scores[3] += 0.2  # Context Coverage
                elif custom_chunk_size < original_chunk_size:
                    custom_scores[0] -= 0.1  # Accuracy
                    custom_scores[1] += 0.1  # Latency
                    custom_scores[2] += 0.1  # Cost
                    custom_scores[3] -= 0.2  # Context Coverage
                
                # Top-k impact
                original_top_k = parse_top_k(parameters.get('top_k', '5'))
                if custom_top_k > original_top_k:
                    custom_scores[0] += 0.1  # Accuracy
                    custom_scores[1] -= 0.2  # Latency
                    custom_scores[2] -= 0.1  # Cost
                    custom_scores[4] += 0.1  # Relevance
                elif custom_top_k < original_top_k:
                    custom_scores[0] -= 0.1  # Accuracy
                    custom_scores[1] += 0.2  # Latency
                    custom_scores[2] += 0.1  # Cost
                    custom_scores[4] -= 0.1  # Relevance
                
                # Ensure scores are within [0, 1]
                custom_scores = [max(0, min(1, score)) for score in custom_scores]
                
                # Create radar chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=original_scores,
                    theta=categories,
                    fill='toself',
                    name='Original Parameters'
                ))
                
                fig.add_trace(go.Scatterpolar(
                    r=custom_scores,
                    theta=categories,
                    fill='toself',
                    name='Custom Parameters'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True
                )
                
                st.plotly_chart(fig)
                
                # Display recommendations based on customization
                st.subheader("Recommendations")
                
                if custom_chunk_size > 2000:
                    st.warning("Large chunk sizes may lead to less precise retrieval. Consider using a reranker to improve relevance.")
                
                if custom_top_k > 10:
                    st.warning("High top-k values may introduce noise. Consider using a reranker to filter irrelevant chunks.")
                
                if custom_overlap < 50:
                    st.warning("Low overlap may cause context loss between chunks. Consider increasing overlap for better context preservation.")
    
    # Chat Interface
    with tab2:
        st.header("Chat-based Parameter Optimization")
        st.write("Chat with our assistant to get RAG parameter recommendations tailored to your needs.")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hi! I'm your RAG parameter optimization assistant. Tell me about your use case, and I'll recommend the best parameters for your RAG system. You can mention details like document type (technical, legal, educational, code), performance priorities (accuracy, latency, cost), and data size (small, medium, large)."}
            ]
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                try:
                    st.markdown(message["content"])
                except Exception as e:
                    st.text(message["content"])
                    st.warning(f"Note: Markdown rendering failed, displaying as plain text.")
        
        # Chat input
        if prompt := st.chat_input("What's your RAG use case?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Extract parameters from chat history
            extracted_params = extract_parameters_from_chat(st.session_state.messages)
            
            # Generate response based on extracted parameters
            with st.chat_message("assistant"):
                if not extracted_params["use_case"]:
                    response = "I need to know your use case to provide specific recommendations. Could you please tell me what you're using RAG for? For example, is it for knowledge management, customer support, healthcare applications, education, code assistance, sales automation, marketing, threat analysis, or gaming?"
                    safe_display(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    # Get optimized parameters
                    parameters = get_rag_parameters(
                        extracted_params["use_case"],
                        extracted_params["document_type"],
                        extracted_params["performance_priority"],
                        extracted_params["data_size"]
                    )
                    
                    # Configure pipeline
                    pipeline_config = configure_rag_pipeline(parameters)
                    
                    # Generate response
                    response = f"Based on your {extracted_params['use_case']} use case"
                    
                    if extracted_params["document_type"] != "general":
                        response += f" with {extracted_params['document_type']} documents"
                    
                    if extracted_params["performance_priority"] != "balanced":
                        response += f", prioritizing {extracted_params['performance_priority']}"
                    
                    if extracted_params["data_size"] != "medium":
                        response += f", and {extracted_params['data_size']} data size"
                    
                    response += ", here are my recommended RAG parameters:\n\n"
                    
                    # Add key parameters
                    response += f"**Chunk Size:** {parameters.get('chunk_size_tokens', 'Not specified')}\n\n"
                    response += f"**Chunk Overlap:** {parameters.get('overlap_tokens', 'Not specified')}\n\n"
                    response += f"**Embedding Model:** {parameters.get('embedding_model_recommendation', 'Not specified')}\n\n"
                    response += f"**Top-K:** {parameters.get('top_k', 'Not specified')}\n\n"
                    response += f"**Retrieval Enhancements:** {parameters.get('retrieval_enhancements', 'Not specified')}\n\n"
                    response += f"**Generation Settings:** {parameters.get('generation_settings', 'Not specified')}\n\n"
                    response += f"**Chunking Strategy:** {parameters.get('chunking_strategy_recommendation', 'Not specified')}\n\n"
                    
                    # Add explanation
                    response += "Would you like me to explain any of these parameters in more detail or adjust them based on additional requirements?"
                    
                    safe_display(response)
                    
                    # Add visualization
                    st.subheader("RAG Pipeline Visualization")
                    
                    # Generate Mermaid diagram
                    mermaid_code = generate_mermaid_diagram(pipeline_config)
                    
                    # Display Mermaid diagram
                    try:
                        # Try to import streamlit-mermaid
                        from streamlit_mermaid import st_mermaid
                        st_mermaid(mermaid_code)
                    except ImportError:
                        st.code(mermaid_code, language="mermaid")
                        st.warning("Install streamlit-mermaid for interactive diagrams: `pip install streamlit-mermaid`")
                    
                    # Add response to chat history (without the visualization)
                    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Llama Stack Testing Interface
    with tab3:
        st.header("Llama Stack RAG Testing")
        st.write("Test your RAG parameters with Llama Stack's RAG tools and VectorIO APIs.")
        
        # Initialize session state for Llama Stack integration
        if "llama_stack_integration" not in st.session_state:
            # Get port and model from environment variables or use defaults
            port = os.environ.get("LLAMA_STACK_PORT", "5002")
            model_id = os.environ.get("INFERENCE_MODEL", "llama3.2:3b-instruct-fp16")
            
            # Initialize Llama Stack integration
            try:
                st.session_state.llama_stack_integration = LlamaStackRAGIntegration(
                    port=port,
                    model_id=model_id
                )
                st.session_state.llama_stack_connected = True
            except Exception as e:
                st.session_state.llama_stack_connected = False
                st.error(f"Failed to connect to Llama Stack: {str(e)}")
        
        # Check if Llama Stack is connected
        if not st.session_state.get("llama_stack_connected", False):
            st.error("Not connected to Llama Stack. Please check your connection settings.")
            
            # Allow user to retry connection
            port = st.text_input("Llama Stack Port", value="5002")
            model_id = st.text_input("Model ID", value="llama3.2:3b-instruct-fp16")
            
            if st.button("Connect to Llama Stack"):
                try:
                    st.session_state.llama_stack_integration = LlamaStackRAGIntegration(
                        port=port,
                        model_id=model_id
                    )
                    st.session_state.llama_stack_connected = True
                    st.success("Successfully connected to Llama Stack!")
                except Exception as e:
                    st.error(f"Failed to connect to Llama Stack: {str(e)}")
        else:
            # Display connection status
            st.success(f"Connected to Llama Stack using model: {st.session_state.llama_stack_integration.model_id}")
            
            # Create columns for configuration and document upload
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("RAG Configuration")
                
                # Use case selection
                use_cases = [
                    "Knowledge Management", 
                    "Customer Support", 
                    "Healthcare Applications", 
                    "Education", 
                    "Code Assistance", 
                    "Sales Automation", 
                    "Marketing", 
                    "Threat Analysis", 
                    "Gaming"
                ]
                use_case = st.selectbox("Select Use Case", use_cases, key="llama_use_case")
                
                # Document type selection
                document_types = ["general", "technical", "legal", "educational", "code"]
                document_type = st.selectbox("Document Type", document_types, key="llama_doc_type")
                
                # Performance priority selection
                performance_priorities = ["balanced", "accuracy", "latency", "cost"]
                performance_priority = st.selectbox("Performance Priority", performance_priorities, key="llama_perf_priority")
                
                # Data size selection
                data_sizes = ["small", "medium", "large"]
                data_size = st.selectbox("Data Size", data_sizes, key="llama_data_size")
                
                # Vector database ID
                vector_db_id = st.text_input("Vector Database ID", value="rag-optimizer-db")
            
            with col2:
                st.subheader("Document Upload")
                
                # File uploader for documents
                uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True, type=["txt", "md", "py", "js", "html", "css", "json", "csv"])
                
                # Display uploaded files
                if uploaded_files:
                    st.write(f"Uploaded {len(uploaded_files)} documents:")
                    for file in uploaded_files:
                        st.write(f"- {file.name} ({file.type}, {file.size} bytes)")
                else:
                    st.info("No documents uploaded. Please upload at least one document to test RAG.")
            
            # Query input
            st.subheader("Query")
            query = st.text_input("Enter your query", value="What is the main topic of these documents?")
            
            # Run RAG button
            run_button = st.button("Run RAG Pipeline")
            
            if run_button:
                if not uploaded_files:
                    st.error("Please upload at least one document to test RAG.")
                else:
                    # Process uploaded files
                    documents = process_uploaded_files(uploaded_files)
                    
                    # Display progress
                    with st.spinner("Running RAG pipeline..."):
                        # Run RAG pipeline
                        result = st.session_state.llama_stack_integration.run_rag_pipeline(
                            use_case=use_case,
                            documents=documents,
                            query=query,
                            document_type=document_type,
                            performance_priority=performance_priority,
                            data_size=data_size,
                            vector_db_id=vector_db_id
                        )
                    
                    # Display results
                    if result["success"]:
                        st.success("RAG pipeline executed successfully!")
                        
                        # Display query and response
                        st.subheader("Query")
                        st.write(result["query"])
                        
                        st.subheader("Response")
                        # Use safe display for the response
                        try:
                            # Try to display as Markdown
                            st.markdown(result["response"])
                        except Exception as e:
                            # If Markdown rendering fails, display as plain text
                            st.text_area("Response (plain text)", result["response"], height=200)
                            st.warning(f"Note: Markdown rendering failed, displaying as plain text.")
                        
                        # Display parameters and configuration
                        with st.expander("Show RAG Parameters"):
                            st.json(result["parameters"])
                        
                        with st.expander("Show Llama Stack Configuration"):
                            st.json({k: v for k, v in result["config"].items() if k != "original_parameters"})
                    else:
                        st.error(f"RAG pipeline execution failed: {result.get('error', 'Unknown error')}")

# Integration with Llama Stack
def integrate_with_llama_stack():
    """
    Function to integrate the RAG parameter optimizer with Llama Stack.
    This would be implemented in a separate file.
    """
    from llama_stack_client.lib.agents.client_tool import client_tool
    
    @client_tool
    def optimize_rag_parameters(
        use_case: str,
        document_type: str = "general",
        performance_priority: str = "balanced",
        data_size: str = "medium"
    ) -> dict:
        """
        Optimizes RAG parameters based on use case and other factors.
        
        :param use_case: The primary use case (e.g., "Knowledge Management", "Customer Support")
        :param document_type: Type of documents (e.g., "technical", "legal", "educational", "code")
        :param performance_priority: Optimization priority (e.g., "accuracy", "latency", "cost", "balanced")
        :param data_size: Size of the data corpus (e.g., "small", "medium", "large")
        :returns: Dictionary of optimized RAG parameters
        """
        # Get optimized parameters
        return get_rag_parameters(use_case, document_type, performance_priority, data_size)
    
    @client_tool
    def configure_rag_pipeline_tool(parameters: dict) -> dict:
        """
        Configures a RAG pipeline based on the provided parameters.
        
        :param parameters: Dictionary of RAG parameters
        :returns: Dictionary containing the configured pipeline components
        """
        # Configure pipeline
        return configure_rag_pipeline(parameters)


if __name__ == "__main__":
    main()
