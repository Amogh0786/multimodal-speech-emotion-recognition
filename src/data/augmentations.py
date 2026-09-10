import torch
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, RoomSimulator
from typing import Optional

class AudioAugmenter:
    """Audio augmentation pipeline using audiomentations."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5), # SNR 10-30dB approx
            TimeStretch(min_rate=0.85, max_rate=1.15, leave_length_unchanged=True, p=0.5),
            PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
            # RoomSimulator requires carefully crafted IRs or defaults, using basic params
            RoomSimulator(p=0.3)
        ])
        
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Applies augmentation to a 1D tensor."""
        # audiomentations expects numpy arrays
        wav_np = waveform.numpy()
        aug_wav_np = self.augment(samples=wav_np, sample_rate=self.sample_rate)
        return torch.from_numpy(aug_wav_np).float()
