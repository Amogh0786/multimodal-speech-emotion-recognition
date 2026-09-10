import librosa
import numpy as np
import torch
from typing import Tuple

class AudioPreprocessor:
    """Standardizes audio to uniform length and sample rate."""
    
    def __init__(self, target_sr: int = 16000, duration: float = 3.0):
        self.target_sr = target_sr
        self.target_length = int(target_sr * duration) # 48000
        
    def process_file(self, file_path: str) -> torch.Tensor:
        """Loads, trims silence, resamples, and pads/chunks to target length."""
        # Load and resample
        wav, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
        
        # Trim silence (VAD based on energy)
        wav, _ = librosa.effects.trim(wav, top_db=25)
        
        # Pad or chunk to exact length
        if len(wav) < self.target_length:
            pad_len = self.target_length - len(wav)
            wav = np.pad(wav, (0, pad_len), mode='constant')
        elif len(wav) > self.target_length:
            # Center crop
            start = (len(wav) - self.target_length) // 2
            wav = wav[start:start + self.target_length]
            
        return torch.from_numpy(wav).float()
