"""Validated configuration for the Artemide decoder-only language model."""
from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 259
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    max_seq_len: int = 512
    dropout: float = 0.0
    ff_mult: int = 4

    def __post_init__(self):
        if self.vocab_size < 2 or self.d_model < 1 or self.n_layers < 1:
            raise ValueError("vocab_size, d_model and n_layers must be positive")
        if self.n_heads < 1 or self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("attention head dimension must be even for rotary embeddings")
        if self.max_seq_len < 1 or self.ff_mult < 1:
            raise ValueError("max_seq_len and ff_mult must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self):
        return asdict(self)
