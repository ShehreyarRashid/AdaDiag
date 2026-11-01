"""AdaDiag — adaptive modality x task weighting for multimodal failure diagnosis."""
from . import config, data, diagnostics, evaluate, losses, model, train, weighting

__version__ = "1.0.0"
__all__ = ["config", "data", "diagnostics", "evaluate", "losses", "model", "train", "weighting"]
