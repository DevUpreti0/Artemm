"""Artemide: independently initialized decoder-only language model."""
__version__ = "1.7.0"
from .model.config import ModelConfig
from .model.transformer import ArtemideLM
from .tokenizer import ByteTokenizer

__all__ = ["ModelConfig", "ArtemideLM", "ByteTokenizer", "__version__"]
