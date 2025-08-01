import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Audio settings
    SAMPLE_RATE = 16000
    MAX_AUDIO_LENGTH = 300  # 5 minutes in seconds
    
    # Transcription settings
    WHISPER_MODEL = WHISPER_MODEL = "openai/whisper-tiny"
    DIARIZATION_MODEL = "pyannote/speaker-diarization"
    
    # Gemini API
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # Hugging Face
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    # Models for analysis
    SENTIMENT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
    TONALITY_MODEL = "bhadresh-savani/bert-base-go-emotion"
    INTENT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    
    # RAG settings
    VECTOR_DB_PATH = "data/vector_db"
    DOCUMENTS_DIR = "data/documents"
    
    # Evaluation
    TEST_DATA_DIR = "data/test_audios"