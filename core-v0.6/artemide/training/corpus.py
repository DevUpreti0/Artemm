"""Deterministic, auditable corpus preparation for byte-level LM training."""
import hashlib, json
from pathlib import Path
import torch
from artemide.tokenizer import ByteTokenizer
from artemide.training.data import load_texts, split_documents

def prepare_corpus(source, output_dir, val_fraction=0.05, seed=42, license_name="unspecified"):
    source, out = Path(source), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    docs = load_texts(source)
    if not docs:
        raise ValueError(f"No usable .txt or .jsonl documents found in {source}")
    train_docs, val_docs = split_documents(docs, val_fraction, seed)
    tok = ByteTokenizer()
    def write_split(name, documents):
        ids = []
        for doc in documents:
            ids.extend(tok.encode(doc, add_bos=True, add_eos=True))
        if len(ids) < 2:
            raise ValueError(f"{name} split has too few tokens")
        target = out / f"{name}.bin"
        torch.tensor(ids, dtype=torch.uint16).numpy().tofile(target)
        return {"file": target.name, "tokens": len(ids),
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    manifest = {"format":"artemide-byte-token-v1","tokenizer":"ByteTokenizer",
        "vocab_size":tok.vocab_size,"seed":seed,"val_fraction":val_fraction,
        "license":license_name,"documents":len(docs),
        "train":write_split("train",train_docs),
        "validation":write_split("validation",val_docs) if val_docs else None}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return manifest
