import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, device: torch.device):
        super().__init__()
        self.layer = nn.Linear(hidden_dim, num_classes, device=device)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.layer(h)