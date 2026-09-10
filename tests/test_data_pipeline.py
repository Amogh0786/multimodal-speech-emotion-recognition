import pytest
import torch
from src.data.preprocessor import AudioPreprocessor
from src.data.augmentations import AudioAugmenter

def test_preprocessor_shape():
    # Mock file or use synthetic tensor if we refactor preprocessor.
    # Since preprocessor currently requires a file path, we would create a dummy wav file here.
    import soundfile as sf
    import numpy as np
    import os
    
    dummy_wav = np.random.randn(32000)
    sf.write("dummy.wav", dummy_wav, 16000)
    
    preprocessor = AudioPreprocessor(target_sr=16000, duration=3.0)
    tensor = preprocessor.process_file("dummy.wav")
    
    os.remove("dummy.wav")
    
    assert tensor.shape == (48000,), "Output shape must be exactly 48000 samples for 3s at 16kHz"

def test_augmentations():
    augmenter = AudioAugmenter(sample_rate=16000)
    tensor = torch.randn(48000)
    aug_tensor = augmenter(tensor)
    
    assert aug_tensor.shape == tensor.shape, "Augmentation should not change shape"
