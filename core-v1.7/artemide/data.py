"""Deterministic corpus preparation and uint16 token stream sampling."""
from __future__ import annotations
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Iterable
import torch
from .tokenizer import ByteTokenizer


def load_documents(source):
    root = Path(source)
    if not root.exists():
        raise FileNotFoundError(root)
    paths = [root] if root.is_file() else sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".jsonl"}
    )
    docs = []
    for path in paths:
        if path.suffix.lower() == ".txt":
            value = path.read_text(encoding="utf-8-sig")
            if value.strip():
                docs.append((path.name if root.is_file() else str(path.relative_to(root)), value))
        elif path.suffix.lower() == ".jsonl":
            with path.open(encoding="utf-8-sig") as handle:
                for number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"invalid JSONL at {path}:{number}") from exc
                    value = record if isinstance(record, str) else (
                        record.get("text") if isinstance(record, dict) else None
                    )
                    if not isinstance(value, str):
                        raise ValueError(f"JSONL record {path}:{number} needs a string text field")
                    if value.strip():
                        docs.append((f"{path.name if root.is_file() else path.relative_to(root)}:{number}", value))
    if not docs:
        raise ValueError("no usable .txt or .jsonl documents found")
    return docs


def split_documents(documents, validation_fraction=0.05, seed=42):
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if not documents:
        raise ValueError("no documents")
    if validation_fraction == 0:
        return list(documents), []
    if len(documents) < 2:
        raise ValueError("at least two documents required for validation split")
    shuffled = list(documents)
    random.Random(seed).shuffle(shuffled)
    count = min(len(shuffled) - 1, max(1, round(len(shuffled) * validation_fraction)))
    return shuffled[count:], shuffled[:count]


def _write_tokens(path, documents, tokenizer):
    digest, count = hashlib.sha256(), 0
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            for _, text in documents:
                for token in tokenizer.encode(text, add_bos=True, add_eos=True):
                    raw = token.to_bytes(2, "little")
                    handle.write(raw)
                    digest.update(raw)
                    count += 1
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)
    if count < 2:
        raise ValueError(f"{path.name} contains too few tokens")
    return {"file": path.name, "tokens": count, "sha256": digest.hexdigest()}


def prepare_corpus(source, output_dir, *, validation_fraction=0.05, seed=42, license_name="unspecified"):
    if not license_name.strip():
        raise ValueError("license_name must be non-empty")
    docs = load_documents(source)
    train_docs, val_docs = split_documents(docs, validation_fraction, seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteTokenizer()
    train = _write_tokens(output / "train.bin", train_docs, tokenizer)
    validation = _write_tokens(output / "validation.bin", val_docs, tokenizer) if val_docs else None
    manifest = {
        "format": "artemide-byte-token-v1", "tokenizer": "ByteTokenizer",
        "vocab_size": tokenizer.vocab_size, "token_storage": "uint16-little-endian",
        "seed": seed, "validation_fraction": validation_fraction,
        "license_declared_by_operator": license_name, "documents_total": len(docs),
        "documents_train": len(train_docs), "documents_validation": len(val_docs),
        "train": train, "validation": validation,
    }
    temp = output / "manifest.json.tmp"
    temp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, output / "manifest.json")
    return manifest


class TokenStream:
    def __init__(self, path, vocab_size):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        size = self.path.stat().st_size
        if size % 2:
            raise ValueError("uint16 token file has odd byte length")
        count = size // 2
        if count < 2:
            raise ValueError("token file needs at least two IDs")
        self.tokens = torch.from_file(str(self.path), shared=False, size=count, dtype=torch.uint16).long()
        if int(self.tokens.max()) >= vocab_size:
            raise ValueError("token ID exceeds configured vocabulary")

    def sample(self, batch_size, sequence_length, device, generator=None):
        max_start = self.tokens.numel() - sequence_length - 1
        if max_start < 0:
            raise ValueError(f"need at least {sequence_length + 1} tokens")
        starts = torch.randint(0, max_start + 1, (batch_size,), generator=generator).tolist()
        batch = torch.stack([self.tokens[s:s + sequence_length + 1] for s in starts])
        return batch[:, :-1].to(device), batch[:, 1:].to(device)
