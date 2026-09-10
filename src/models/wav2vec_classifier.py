import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
from .temporal_pooling import AttentionPooling

class Wav2Vec2Classifier(nn.Module):
    """Self-Supervised Backbone (Primary) with Temporal Pooling and MLP head."""
    
    def __init__(
        self, 
        model_name: str = "facebook/wav2vec2-base", 
        n_classes: int = 4, 
        freeze_feature_extractor: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        
        if freeze_feature_extractor:
            self.wav2vec2.feature_extractor._freeze_parameters()
            
        hidden_size = self.wav2vec2.config.hidden_size
        
        self.pooling = AttentionPooling(hidden_size)
        
        # Multi-layer Perceptron (MLP) classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes)
        )
        
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (batch_size, seq_length)
        """
        outputs = self.wav2vec2(x, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state # (batch, seq_len, hidden_size)
        
        # Using temporal pooling
        pooled_output = self.pooling(hidden_states, mask=None) # We can pass actual mask if needed
        
        logits = self.classifier(pooled_output)
        
        return logits
