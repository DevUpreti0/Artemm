import json
from pathlib import Path

import pytest

from artemide_tokenizer import ByteTokenizer
from artemide_prepare_corpus import prepare_corpus, split_documents


def test_byte_tokenizer_round_trip_unicode_and_specials():
    tokenizer = ByteTokenizer()
    text = "Artemide — नमस्ते 🌻"
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    assert ids[0] == tokenizer.bos_token_id
    assert ids[-1] == tokenizer.eos_token_id
    assert tokenizer.decode(ids) == text
    assert tokenizer.decode(ids, skip_special_tokens=False).startswith("<bos>")


def test_byte_tokenizer_rejects_invalid_ids():
    with pytest.raises(ValueError, match="outside vocabulary"):
        ByteTokenizer().decode([259])


def test_document_split_is_deterministic_and_disjoint():
    docs = [(f"doc-{i}", f"text {i}") for i in range(20)]
    train_a, val_a = split_documents(docs, 0.2, 17)
    train_b, val_b = split_documents(docs, 0.2, 17)
    assert train_a == train_b
    assert val_a == val_b
    assert set(x[0] for x in train_a).isdisjoint(x[0] for x in val_a)


def test_prepare_corpus_writes_compatible_uint16_files(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "one.txt").write_text("hello", encoding="utf-8")
    (source / "two.jsonl").write_text(
        json.dumps({"text": "नमस्ते"}) + "\n", encoding="utf-8"
    )
    out = tmp_path / "prepared"
    manifest = prepare_corpus(
        source, out, validation_fraction=0.5, seed=3, license_name="CC0 test fixture"
    )
    assert manifest["vocab_size"] == 259
    assert manifest["documents_total"] == 2
    assert (out / "train.bin").stat().st_size == manifest["train"]["tokens"] * 2
    assert (out / "validation.bin").stat().st_size == manifest["validation"]["tokens"] * 2
    assert json.loads((out / "manifest.json").read_text())["tokenizer"] == "ByteTokenizer"


def test_validation_split_requires_multiple_documents():
    with pytest.raises(ValueError, match="at least two documents"):
        split_documents([("only", "one")], 0.1, 42)
