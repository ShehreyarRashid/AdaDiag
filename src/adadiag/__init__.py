"""AdaDiag — adaptive modality x task weighting for multimodal failure diagnosis."""
from . import (config, data, diagnostics, evaluate, identifiability, losses, model, train,
               weighting)

__version__ = "1.1.0"
__all__ = ["config", "data", "diagnostics", "evaluate", "identifiability", "losses",
           "model", "train", "weighting"]
