# Conversational RAG Assistant

A voice-enabled conversational assistant that uses Retrieval Augmented Generation (RAG) with Llama Stack and ElevenLabs for text-to-speech.

## Features

- Voice-based conversational interface
- Document ingestion and retrieval using RAG
- Powered by Llama Stack and Llama 3.3 70B model
- Text-to-speech using ElevenLabs

## Setup

### Prerequisites

- Python 3.8+
- Llama Stack server running locally
- ElevenLabs API key
- Required Python packages (see requirements.txt)

### Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with the following variables:
   ```
   LLAMA_STACK_PORT=8000
   ELEVENLABS_API_KEY=your_elevenlabs_api_key
   ELEVENLABS_VOICE_ID=your_preferred_voice_id
   ```

### Adding Documents

1. Create a `documents` folder in the project directory
2. Add text files (.txt) containing the knowledge you want the assistant to access
3. The assistant will automatically ingest these documents on startup

## Usage

1. Start the Llama Stack server (follow Llama Stack documentation)
2. Run the assistant:
   ```bash
   python rag_assistant.py
   ```
3. The assistant will:
   - Set up a vector database
   - Ingest documents from the `documents` folder
   - Create a conversational agent
   - Greet you with voice
4. Type your questions and the assistant will:
   - Search the knowledge base for relevant information
   - Provide answers based on the documents
   - Speak the responses using ElevenLabs

## How It Works

1. **Document Ingestion**: Text files are chunked and stored in a vector database
2. **RAG**: When you ask a question, the system retrieves relevant document chunks
3. **LLM Processing**: Llama 3.3 70B generates responses based on the retrieved context
4. **Voice Synthesis**: ElevenLabs converts the text response to natural-sounding speech

## Customization

- Modify the agent instructions in `create_agent()` function
- Adjust chunking parameters in `ingest_documents()` function
- Change the voice settings in `play_audio()` function

## Troubleshooting

- Ensure Llama Stack server is running on the specified port
- Check that your ElevenLabs API key is valid
- Verify that document files are readable text files (.txt)
- Make sure all dependencies are installed correctly