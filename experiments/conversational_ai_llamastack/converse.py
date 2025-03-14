import os
import time
import base64
from pathlib import Path
import requests
from dotenv import load_dotenv
import sounddevice as sd
import soundfile as sf
from llama_stack_client import LlamaStackClient
from llama_stack_client.types import Document
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger

# Try to import speech recognition libraries
# If not available, voice commands will be disabled
VOICE_COMMANDS_AVAILABLE = False
try:
    import speech_recognition as sr
    VOICE_COMMANDS_AVAILABLE = True
except ImportError:
    print("Speech recognition libraries not available. Voice commands will be disabled.")
    print("To enable voice commands, install the required dependencies:")
    print("1. Install PortAudio: brew install portaudio")
    print("2. Install Python packages: pip install SpeechRecognition PyAudio")

# Load environment variables
load_dotenv()

# Initialize clients
llama_client = LlamaStackClient(base_url=f"http://localhost:{os.environ.get('LLAMA_STACK_PORT', '8000')}")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Default voice

def setup_vector_db(vector_db_id="knowledge_base"):
    """Set up and register a vector database"""
    try:
        # Check if vector DB already exists
        existing_dbs = [db.identifier for db in llama_client.vector_dbs.list()]
        if vector_db_id in existing_dbs:
            print(f"Vector DB '{vector_db_id}' already exists.")
            return vector_db_id
        
        # Register a new vector DB
        response = llama_client.vector_dbs.register(
            vector_db_id=vector_db_id,
            embedding_model="all-MiniLM-L6-v2",
            embedding_dimension=384,
            provider_id="faiss",
        )
        print(f"Vector DB '{vector_db_id}' registered successfully.")
        return vector_db_id
    except Exception as e:
        print(f"Error setting up vector DB: {e}")
        return None

def ingest_documents(vector_db_id, documents_path="documents"):
    """Ingest documents into the vector database"""
    try:
        # Create documents directory if it doesn't exist
        Path(documents_path).mkdir(exist_ok=True)
        
        # Get list of documents in the directory
        doc_files = list(Path(documents_path).glob("*.txt"))
        if not doc_files:
            print(f"No documents found in {documents_path}. Please add some .txt files.")
            return False
        
        # Prepare documents for ingestion
        documents = []
        for i, doc_file in enumerate(doc_files):
            print(f"Ingesting document {doc_file.name}")
            with open(doc_file, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append(
                    Document(
                        document_id=f"doc-{i}",
                        content=content,
                        mime_type="text/plain",
                        metadata={"filename": doc_file.name}
                    )
                )
        
        # Ingest documents using RAG tool
        llama_client.tool_runtime.rag_tool.insert(
            documents=documents,
            vector_db_id=vector_db_id,
            chunk_size_in_tokens=512,
        )
        
        print(f"Successfully ingested {len(documents)} documents into vector DB.")
        return True
    except Exception as e:
        print(f"Error ingesting documents: {e}")
        return False

def create_agent(vector_db_id):
    """Create a RAG-enabled agent"""
    try:
        model = os.environ.get('INFERENCE_MODEL', "granite3-dense:latest")
        agent = Agent(
            llama_client,
            model=model,
            instructions="""You are a helpful conversational assistant with access to a knowledge base.
            When answering questions, use information from the knowledge base when relevant.
            Keep your responses concise, informative, and conversational.
            If you don't know the answer or can't find relevant information, be honest about it.""",
            tools=[
                {
                    "name": "builtin::rag/knowledge_search",
                    "args": {
                        "vector_db_ids": [vector_db_id],
                    },
                }
            ],
        )
        session_id = agent.create_session("rag_conversation")
        return agent, session_id
    except Exception as e:
        print(f"Error creating agent: {e}")
        return None, None

def listen_for_voice_command():
    """Record audio from microphone and convert to text"""
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("\n🎤 Listening... (speak now)")
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            # Record audio
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            print("🔍 Processing speech...")
            
            # Convert speech to text
            text = recognizer.recognize_google(audio)
            print(f"🔊 You said: {text}")
            return text
    except sr.WaitTimeoutError:
        print("No speech detected within timeout period.")
        return None
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return None
    except Exception as e:
        print(f"Error in speech recognition: {e}")
        return None

def play_audio(text):
    """Convert text to speech using ElevenLabs and play it"""
    try:
        # ElevenLabs API call
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY
        }
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            # Save audio to file
            audio_dir = Path("audio_responses")
            audio_dir.mkdir(exist_ok=True)
            timestamp = int(time.time())
            audio_path = audio_dir / f"response_{timestamp}.mp3"
            
            with open(audio_path, "wb") as f:
                f.write(response.content)
            
            # Play audio
            print("🔊 Playing audio response...")
            data, samplerate = sf.read(audio_path)
            sd.play(data, samplerate)
            sd.wait()
            return True
        else:
            print(f"Error from ElevenLabs API: {response.status_code}, {response.text}")
            return False
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
        return False

def main():
    """Main function to run the conversational RAG assistant"""
    print("🤖 Setting up Conversational RAG Assistant...")
    
    # Setup vector database
    vector_db_id = setup_vector_db()
    if not vector_db_id:
        print("Failed to set up vector database. Exiting.")
        return
    
    # Ingest documents
    success = ingest_documents(vector_db_id)
    if not success:
        print("No documents ingested. The assistant will have limited knowledge.")
    
    # Create agent
    agent, session_id = create_agent(vector_db_id)
    if not agent or not session_id:
        print("Failed to create agent. Exiting.")
        return
    
    print("\n🎙️ Conversational RAG Assistant is ready!")
    
    # Show appropriate instructions based on voice command availability
    if VOICE_COMMANDS_AVAILABLE:
        print("Type your questions, say 'voice' to use voice input, or 'exit' to quit.")
        greeting = "Hello! I'm your conversational assistant. I can answer questions based on the documents in my knowledge base. You can type or use voice commands. How can I help you today?"
    else:
        print("Type your questions or 'exit' to quit.")
        greeting = "Hello! I'm your conversational assistant. I can answer questions based on the documents in my knowledge base. How can I help you today?"
    print(f"\nAssistant: {greeting}")
    play_audio(greeting)
    
    # Conversation loop
    conversation_history = []
    voice_mode = False
    
    while True:
        try:
            # Get user input (text or voice)
            if voice_mode:
                user_input = listen_for_voice_command()
                if user_input is None:
                    print("I didn't catch that. Please try again or type 'text' to switch to text input.")
                    continue
            else:
                user_input = input("\nYou: ")
            
            # Handle mode switching commands
            if user_input and user_input.lower() == "voice":
                if VOICE_COMMANDS_AVAILABLE:
                    voice_mode = True
                    print("Switched to voice input mode. Say 'text' to switch back to text input.")
                else:
                    print("Voice commands are not available. Please install the required dependencies:")
                    print("1. Install PortAudio: brew install portaudio")
                    print("2. Install Python packages: pip install SpeechRecognition PyAudio")
                continue
            elif user_input and user_input.lower() == "text":
                voice_mode = False
                print("Switched to text input mode. Type 'voice' to switch back to voice input.")
                continue
            elif user_input and user_input.lower() in ["exit", "quit", "bye"]:
                farewell = "Thank you for chatting with me. Goodbye!"
                print(f"\nAssistant: {farewell}")
                play_audio(farewell)
                break
            
            # Skip empty input
            if not user_input:
                continue
            
            # Add user message to conversation history
            conversation_history.append({"role": "user", "content": user_input})
            
            # Get response from agent
            print("\nThinking...")
            response = agent.create_turn(
                messages=conversation_history,
                session_id=session_id,
            )
            
            # Extract assistant's response using EventLogger
            assistant_message = None
            for log in EventLogger().log(response):
                if hasattr(log, 'message') and log.message.role == "assistant":
                    assistant_message = log.message.content
                    break
            
            if assistant_message:
                print(f"\nAssistant: {assistant_message}")
                
                # Add assistant message to conversation history
                conversation_history.append({"role": "assistant", "content": assistant_message})
                
                # Convert to speech and play
                play_audio(assistant_message)
            else:
                print("No response received from assistant.")
            
            # Keep conversation history manageable
            if len(conversation_history) > 10:
                # Keep the last 10 messages
                conversation_history = conversation_history[-10:]
                
        except KeyboardInterrupt:
            print("\nGracefully shutting down...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
