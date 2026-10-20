"""A learned, bounded neural gate over an own forecast and causal count memory."""
import torch
from torch import nn


class MemoryCountGate(nn.Module):
    def __init__(self, confidence=5., cap=.95):
        super().__init__()
        self.strength=nn.Parameter(torch.tensor(-2.,dtype=torch.float64))
        self.confidence=confidence;self.cap=cap

    def forward(self,parent,mean_log,count,initial):
        gate=self.cap*torch.sigmoid(self.strength)*count/(count+self.confidence)
        blended=torch.expm1((1-gate)*torch.log1p(parent)+gate*mean_log)
        return torch.where(initial & (count>0),blended,parent)
