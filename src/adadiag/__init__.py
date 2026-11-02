"""AdaDiag — adaptive modality x task weighting for multimodal failure diagnosis."""
from . import (ablation, config, data, diagnostics, evaluate, identifiability, losses, model,
               train, weighting)

__version__ = "1.2.0"
__all__ = ["ablation", "config", "data", "diagnostics", "evaluate", "identifiability",
           "losses", "model", "train", "weighting"]
