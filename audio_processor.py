import librosa
import numpy as np
from config import Config

class AudioProcessor:
    def __init__(self):
        self.config = Config()
    
    def load_audio(self, file_path):
        """Load and preprocess audio file"""
        try:
            audio, sr = librosa.load(
                file_path, 
                sr=self.config.SAMPLE_RATE,
                duration=self.config.MAX_AUDIO_LENGTH
            )
            return audio, sr
        except Exception as e:
            raise Exception(f"Error loading audio: {str(e)}")
    
    def normalize_audio(self, audio):
        """Normalize audio to -1 to 1 range"""
        return librosa.util.normalize(audio)
    
    def remove_silence(self, audio, top_db=20):
        """Remove silent segments from audio"""
        non_silent_intervals = librosa.effects.split(audio, top_db=top_db)
        non_silent_audio = np.concatenate([audio[start:end] for start, end in non_silent_intervals])
        return non_silent_audio