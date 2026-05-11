import torch
import torch.nn as nn
import torch.nn.functional as F


class ProbeModelWrapper(nn.Module):
    """Wraps a probe model to use Trainer."""
    def __init__(self, probe): 
        super().__init__()
        self.probe = probe


    def forward(self, labels=None, **kwargs):
        kwargs.pop("labels", None)
        logits = self.probe(**kwargs)
        if isinstance(logits, dict): logits = logits["logits"]
        out = {"logits": logits}
        if labels is not None: out["loss"] = F.cross_entropy(logits, labels)
        return out
