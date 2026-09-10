import torch
import torch.nn as nn
import torchaudio
import torchvision.models as models

class ResNet18SpectrogramExtractor(nn.Module):
    """2D Spectrogram Baseline (Secondary)."""
    
    def __init__(self, n_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=1024, hop_length=512, n_mels=128
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
        
        # SpecAugment
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=35)
        
        # ResNet18 backbone
        self.backbone = models.resnet18(pretrained=True)
        # Modify first conv to accept 1 channel (spectrogram) instead of 3 (RGB)
        # Or, we can duplicate the channel to 3. Let's use 1 channel to be efficient.
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes)
        )

    def forward(self, x: torch.Tensor, augment: bool = False) -> torch.Tensor:
        """x: (batch, time)"""
        # Mel-spec
        mel = self.melspec(x) # (batch, n_mels, time_frames)
        mel_db = self.amplitude_to_db(mel)
        
        if augment and self.training:
            mel_db = self.freq_mask(mel_db)
            mel_db = self.time_mask(mel_db)
            
        # Add channel dim
        mel_db = mel_db.unsqueeze(1) # (batch, 1, n_mels, time_frames)
        
        # Extract features
        features = self.backbone(mel_db) # (batch, num_ftrs)
        
        # Classify
        logits = self.classifier(features)
        
        return logits
