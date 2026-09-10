import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    """Attention-based Temporal Pooling Layer.
    
    Computes:
    e_t = v^T tanh(W h_t + b)
    alpha_t = softmax(e_t)
    z = sum_t (alpha_t * h_t)
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            mask: (batch_size, seq_len) boolean mask where True means padding
        Returns:
            pooled_output: (batch_size, hidden_size)
        """
        # e_t: (batch_size, seq_len, 1)
        e_t = self.v(torch.tanh(self.W(hidden_states)))
        
        if mask is not None:
            e_t = e_t.masked_fill(mask.unsqueeze(-1), float('-inf'))
            
        alpha_t = F.softmax(e_t, dim=1)
        
        # z: (batch_size, hidden_size)
        z = torch.sum(alpha_t * hidden_states, dim=1)
        
        return z
