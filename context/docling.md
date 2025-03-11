Below is a reorganized version of the content you provided, structured for clarity and logical flow. I've grouped related sections, removed redundant or extraneous details, and streamlined the tutorial into a more cohesive format while preserving all essential information. The revised structure focuses on readability and usability, especially for a technical audience following a tutorial.
Build an AI-Powered Multimodal RAG System with Docling and Granite
Published: 26 February 2025
Authors:  
BJ Hargrave, Open Source Developer, STSM  
Erika Russi, Data Scientist, IBM
Introduction
In this tutorial, you’ll learn to build an AI-powered multimodal retrieval-augmented generation (RAG) system using IBM's Docling and open-source IBM Granite models (vision, embeddings, and generative AI). This system processes unstructured data from PDFs—including text, tables, and images—to answer real-time user queries.  
The tutorial leverages:  
Docling: For document parsing and conversion.  
Granite: For natural language processing (NLP) and vision-to-text generation.  
LangChain: For workflow orchestration and integration.
Available as a Jupyter Notebook in the IBM Granite Community's Granite Snack Cookbook GitHub.
What is Multimodal RAG?
Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by connecting them to external knowledge bases without fine-tuning. Traditional RAG is text-only, while multimodal RAG extends this to diverse data types (e.g., text, images, audio) using multimodal LLMs (MLLMs) like Granite, Google’s Gemini, or OpenAI’s GPT-4o.  
Here, we use Granite to process a PDF’s text and images, enabling a robust query-answering system.
Tutorial Overview
Video: Build an AI-Powered Multimodal RAG System (6:28 min) (link-to-video)  
What You’ll Learn:
Document Preprocessing: Parse PDFs with Docling and generate image descriptions with Granite’s vision model.  
Vector Database Setup: Store processed data for efficient retrieval.  
RAG Pipeline: Connect Granite to the knowledge base for query responses.  
LangChain Integration: Streamline workflows and component interactions.
Outcomes:
Proficiency in document preprocessing, chunking, and image understanding.  
Ability to integrate vector databases for retrieval.  
Skills to apply RAG for real-world use cases.
Prerequisites:
Familiarity with Python.  
Basic knowledge of LLMs, NLP, and computer vision.
Step-by-Step Guide
Step 1: Set Up the Environment
Ensure you’re using Python 3.10, 3.11, or 3.12 in a fresh virtual environment.
python
import sys
assert sys.version_info >= (3, 10) and sys.version_info < (3, 13), "Use Python 3.10, 3.11, or 3.12."
Step 2: Install Dependencies
Install required libraries:
bash
pip install "git+https://github.com/ibm-granite-community/utils.git" \
    transformers \
    pillow \
    langchain_community \
    langchain_huggingface \
    langchain_milvus \
    docling \
    replicate
Step 3: Select AI Models
Configure Logging (Optional)
python
import logging
logging.basicConfig(level=logging.INFO)
Load Granite Models
Embeddings Model (Text-to-Vector):
python
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer

embeddings_model_path = "ibm-granite/granite-embedding-30m-english"
embeddings_model = HuggingFaceEmbeddings(model_name=embeddings_model_path)
embeddings_tokenizer = AutoTokenizer.from_pretrained(embeddings_model_path)
Vision Model (Image-to-Text):
Requires a Replicate API token.
python
from ibm_granite_community.notebook_utils import get_env_var
from langchain_community.llms import Replicate
from transformers import AutoProcessor

vision_model_path = "ibm-granite/granite-vision-3.2-2b"
vision_model = Replicate(
    model=vision_model_path,
    replicate_api_token=get_env_var("REPLICATE_API_TOKEN"),
    model_kwargs={"max_tokens": embeddings_tokenizer.max_len_single_sentence, "min_tokens": 100}
)
vision_processor = AutoProcessor.from_pretrained(vision_model_path)
Generative Model (RAG Response):
python
model_path = "ibm-granite/granite-3.2-8b-instruct"
model = Replicate(
    model=model_path,
    replicate_api_token=get_env_var("REPLICATE_API_TOKEN"),
    model_kwargs={"max_tokens": 1000, "min_tokens": 100}
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
Note: We recommend Granite 3.2 over 3.1 (used in the video) for better performance.
Step 4: Prepare Documents for the Vector Database
Convert PDFs with Docling
Download and process a sample PDF:
python
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

pdf_pipeline_options = PdfPipelineOptions(do_ocr=False, generate_picture_images=True)
format_options = {InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options)}
converter = DocumentConverter(format_options=format_options)

sources = ["https://midwestfoodbank.org/images/AR_2020_WEB2.pdf"]
conversions = {source: converter.convert(source=source).document for source in sources}
Process Text Chunks
Chunk text into manageable sizes:
python
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.types.doc.document import TableItem
from langchain_core.documents import Document

doc_id = 0
texts = []
for source, docling_document in conversions.items():
    for chunk in HybridChunker(tokenizer=embeddings_tokenizer).chunk(docling_document):
        items = chunk.meta.doc_items
        if len(items) == 1 and isinstance(items[0], TableItem):
            continue  # Skip tables for now
        refs = " ".join(map(lambda item: item.get_ref().cref, items))
        print(refs)
        document = Document(
            page_content=chunk.text,
            metadata={"doc_id": (doc_id := doc_id + 1), "source": source, "ref": refs}
        )
        texts.append(document)
print(f"{len(texts)} text document chunks created")
Process Tables
Convert tables to markdown:
python
from docling_core.types.doc.labels import DocItemLabel

doc_id = len(texts)
tables = []
for source, docling_document in conversions.items():
    for table in docling_document.tables:
        if table.label in [DocItemLabel.TABLE]:
            ref = table.get_ref().cref
            print(ref)
            text = table.export_to_markdown()
            document = Document(
                page_content=text,
                metadata={"doc_id": (doc_id := doc_id + 1), "source": source, "ref": ref}
            )
            tables.append(document)
print(f"{len(tables)} table documents created")
Process Images
Generate descriptions with the vision model:
python
import base64
import io
import PIL.Image
import PIL.ImageOps
from IPython.display import display

def encode_image(image: PIL.Image.Image, format: str = "png") -> str:
    image = PIL.ImageOps.exif_transpose(image).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format)
    return f"data:image/{format};base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"

image_prompt = "If the image contains text, explain the text in the image."
conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": image_prompt}]}]
vision_prompt = vision_processor.apply_chat_template(conversation=conversation, add_generation_prompt=True)

pictures = []
doc_id = len(texts) + len(tables)
for source, docling_document in conversions.items():
    for picture in docling_document.pictures:
        ref = picture.get_ref().cref
        print(ref)
        image = picture.get_image(docling_document)
        if image:
            text = vision_model.invoke(vision_prompt, image=encode_image(image))
            document = Document(
                page_content=text,
                metadata={"doc_id": (doc_id := doc_id + 1), "source": source, "ref": ref}
            )
            pictures.append(document)
print(f"{len(pictures)} image descriptions created")
Display Processed Documents (Optional)
python
import itertools

for document in itertools.chain(texts, tables):
    print(f"Document ID: {document.metadata['doc_id']}\nSource: {document.metadata['source']}\nContent:\n{document.page_content}\n{'=' * 80}")
for document in pictures:
    print(f"Document ID: {document.metadata['doc_id']}\nSource: {document.metadata['source']}\nContent:\n{document.page_content}")
    image = RefItem(cref=document.metadata['ref']).resolve(conversions[document.metadata['source']]).get_image(conversions[document.metadata['source']])
    print("Image:")
    display(image)
    print("=" * 80)
Populate the Vector Database
Use Milvus to store embeddings:
python
import tempfile
from langchain_milvus import Milvus

db_file = tempfile.NamedTemporaryFile(prefix="vectorstore_", suffix=".db", delete=False).name
print(f"The vector database will be saved to {db_file}")

vector_db = Milvus(
    embedding_function=embeddings_model,
    connection_args={"uri": db_file},
    auto_id=True,
    enable_dynamic_field=True,
    index_params={"index_type": "AUTOINDEX"}
)

documents = list(itertools.chain(texts, tables, pictures))
ids = vector_db.add_documents(documents)
print(f"{len(ids)} documents added to the vector database")
Step 5: Build the RAG Pipeline with Granite
Test Retrieval
Search the vector database:
python
query = "How much was spent on food distribution relative to the amount of food distributed?"
for doc in vector_db.as_retriever().invoke(query):
    print(doc)
    print("=" * 80)
Create the RAG Pipeline
Set up prompts and chain:
python
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

prompt = tokenizer.apply_chat_template(
    conversation=[{"role": "user", "content": "{input}"}],
    documents=[{"title": "placeholder", "text": "{context}"}],
    add_generation_prompt=True,
    tokenize=False
)
prompt_template = PromptTemplate.from_template(template=prompt)
document_prompt_template = PromptTemplate.from_template(template="Document {doc_id}\n{page_content}")
document_separator = "\n\n"

combine_docs_chain = create_stuff_documents_chain(
    llm=model,
    prompt=prompt_template,
    document_prompt=document_prompt_template,
    document_separator=document_separator
)
rag_chain = create_retrieval_chain(retriever=vector_db.as_retriever(), combine_docs_chain=combine_docs_chain)
Generate a Response
python
outputs = rag_chain.invoke({"input": query})
print(outputs['answer'])
Result: Your AI system now leverages text and image data to answer queries!
Next Steps
Explore advanced RAG workflows for other industries.  
Experiment with diverse document types and larger datasets.  
Optimize prompts for improved Granite responses.
Additional Resources
IBM watsonx.ai: Build AI applications efficiently. Explore watsonx.ai (link)  
Granite Models: Learn about IBM’s open-source AI family. Meet Granite (link)  
AI Academy: Free courses for business leaders. Start Learning (link)  
Report: AI in Action 2024 – Insights from 2,000 organizations. Read Now (link)
About IBM
Solutions: AI, consulting, and industry-specific offerings. Explore (link)  
Community: Join the IBM TechXChange (link) or follow on LinkedIn (link), X (link), YouTube (link).  
Careers: Work with IBM (link)
This reorganized version enhances readability by grouping setup steps, code blocks, and explanations logically, while trimming unrelated IBM promotional content to an "Additional Resources" section. Let me know if you'd like further refinements!