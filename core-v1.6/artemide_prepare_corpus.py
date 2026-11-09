"""Prepare licensed text documents into deterministic Artemide uint16 token files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Iterable

from artemide_tokenizer import ByteTokenizer


def load_documents(source: str | Path) -> list[tuple[str, str]]:
    """Load .txt files as documents and .jsonl text records as separate documents."""
    root = Path(source)
    if not root.exists():
        raise FileNotFoundError(root)
    paths = [root] if root.is_file() else sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".jsonl"}
    )
    documents: list[tuple[str, str]] = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix not in {".txt", ".jsonl"}:
            continue
        if suffix == ".txt":
            text = path.read_text(encoding="utf-8-sig")
            if text.strip():
                documents.append((path.name if root.is_file() else str(path.relative_to(root)), text))
            continue
        with path.open(encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc.msg}") from exc
                if isinstance(record, str):
                    text = record
                elif isinstance(record, dict) and isinstance(record.get("text"), str):
                    text = record["text"]
                else:
                    raise ValueError(
                        f"JSONL record at {path}:{line_number} must be a string "
                        "or an object with a string 'text' field"
                    )
                if text.strip():
                    relative = path.name if root.is_file() else str(path.relative_to(root))
                    documents.append((f"{relative}:{line_number}", text))
    return documents


def split_documents(
    documents: list[tuple[str, str]], validation_fraction: float, seed: int
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if not documents:
        raise ValueError("no usable .txt or .jsonl text documents found")
    if validation_fraction == 0:
        return list(documents), []
    if len(documents) < 2:
        raise ValueError("at least two documents are required for a leakage-safe validation split")
    shuffled = list(documents)
    random.Random(seed).shuffle(shuffled)
    validation_count = min(len(shuffled) - 1, max(1, round(len(shuffled) * validation_fraction)))
    return shuffled[validation_count:], shuffled[:validation_count]


def _write_tokens(path: Path, documents: Iterable[tuple[str, str]], tokenizer: ByteTokenizer):
    token_count = 0
    digest = hashlib.sha256()
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            for _name, text in documents:
                ids = tokenizer.encode(text, add_bos=True, add_eos=True)
                # Stream little-endian uint16 without requiring NumPy.
                for token_id in ids:
                    encoded = token_id.to_bytes(2, byteorder="little", signed=False)
                    handle.write(encoded)
                    digest.update(encoded)
                    token_count += 1
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)
    if token_count < 2:
        raise ValueError(f"{path.name} contains too few tokens")
    return {"file": path.name, "tokens": token_count, "sha256": digest.hexdigest()}


def prepare_corpus(
    source: str | Path,
    output_dir: str | Path,
    *,
    validation_fraction: float = 0.05,
    seed: int = 42,
    license_name: str = "unspecified",
) -> dict:
    """Prepare train.bin, optional validation.bin, and a reproducibility manifest."""
    if not license_name.strip():
        raise ValueError("license_name must be a non-empty description")
    docs = load_documents(source)
    train_docs, val_docs = split_documents(docs, validation_fraction, seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteTokenizer()
    train_info = _write_tokens(out / "train.bin", train_docs, tokenizer)
    val_info = _write_tokens(out / "validation.bin", val_docs, tokenizer) if val_docs else None
    manifest = {
        "format": "artemide-byte-token-v1",
        "tokenizer": "ByteTokenizer",
        "vocab_size": tokenizer.vocab_size,
        "special_tokens": {"pad": 0, "bos": 1, "eos": 2},
        "token_storage": "uint16-little-endian",
        "seed": seed,
        "validation_fraction": validation_fraction,
        "license_declared_by_operator": license_name,
        "documents_total": len(docs),
        "documents_train": len(train_docs),
        "documents_validation": len(val_docs),
        "train": train_info,
        "validation": val_info,
    }
    manifest_path = out / "manifest.json"
    temporary_manifest = out / "manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare text for Artemide byte-token training")
    parser.add_argument("--input", required=True, help="directory or .txt/.jsonl file")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--license", default="unspecified", dest="license_name")
    args = parser.parse_args()
    if not 0.0 <= args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be in [0, 1)")
    result = prepare_corpus(
        args.input, args.output,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        license_name=args.license_name,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
