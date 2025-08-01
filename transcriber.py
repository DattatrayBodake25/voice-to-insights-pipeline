from transformers import pipeline
from pyannote.audio import Pipeline
from config import Config
import torch
import librosa
import numpy as np
import json

class Transcriber:
    def __init__(self):
        self.config = Config()
        self._init_models()
    
    def _init_models(self):
        """Initialize Whisper and diarization models"""
        device = 0 if torch.cuda.is_available() else -1  # Fixed device mapping
        
        self.stt_pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.config.WHISPER_MODEL,
            device=device
        )
        
        try:
            self.diarization_pipeline = Pipeline.from_pretrained(
                self.config.DIARIZATION_MODEL,
                use_auth_token=self.config.HF_TOKEN
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load diarization model: {e}")
    
    def transcribe_with_diarization(self, audio_path):
        """Transcribe audio with speaker identification"""
        try:
            # Step 1: Speaker diarization
            diarization = self.diarization_pipeline(audio_path)
            
            # Step 2: Load audio
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Step 3: Loop through each speaker turn
            transcriptions = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start = int(turn.start * sr)
                end = int(turn.end * sr)
                segment = audio[start:end]
                
                result = self.stt_pipeline({
                    "array": np.array(segment),
                    "sampling_rate": sr
                })
                
                transcriptions.append({
                    "speaker": speaker,
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "text": result["text"]
                })
            
            return transcriptions
        
        except Exception as e:
            raise Exception(f"Transcription error: {str(e)}")
    
    def save_transcription(self, transcriptions, output_path):
        """Save transcription to JSON file"""
        with open(output_path, "w") as f:
            json.dump(transcriptions, f, indent=2)