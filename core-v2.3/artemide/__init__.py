"""Artemide independent language-model research stack."""
__version__ = "2.3.0"
from .model.config import ModelConfig
from .model.transformer import ArtemideLM
from .tokenizer import ByteTokenizer
__all__ = ["ModelConfig", "ArtemideLM", "ByteTokenizer", "__version__"]
