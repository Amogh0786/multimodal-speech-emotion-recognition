import torch
from src.models.wav2vec_classifier import Wav2Vec2Classifier
from src.models.feature_extractors import ResNet18SpectrogramExtractor

def test_wav2vec2_forward():
    model = Wav2Vec2Classifier(n_classes=4)
    # Batch size 2, 3 seconds at 16kHz
    dummy_input = torch.randn(2, 48000)
    
    logits = model(dummy_input)
    assert logits.shape == (2, 4), "Logits shape should be (batch_size, n_classes)"
    
def test_resnet_forward():
    model = ResNet18SpectrogramExtractor(n_classes=4)
    dummy_input = torch.randn(2, 48000)
    
    logits = model(dummy_input)
    assert logits.shape == (2, 4), "Logits shape should be (batch_size, n_classes)"
