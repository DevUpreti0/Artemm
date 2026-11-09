"""UTF-8 byte tokenizer: PAD=0, BOS=1, EOS=2, bytes=3..258."""
class ByteTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    vocab_size = 259

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        ids = [byte + 3 for byte in text.encode("utf-8")]
        if add_bos:
            ids.insert(0, self.bos_token_id)
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, token_ids, *, skip_special_tokens: bool = True) -> str:
        output, pieces = bytearray(), []
        def flush():
            if output:
                pieces.append(output.decode("utf-8", errors="replace"))
                output.clear()
        for raw_id in token_ids:
            token_id = int(raw_id)
            if 3 <= token_id < self.vocab_size:
                output.append(token_id - 3)
            elif token_id in (0, 1, 2):
                if not skip_special_tokens:
                    flush()
                    pieces.append({0: "<pad>", 1: "<bos>", 2: "<eos>"}[token_id])
            else:
                raise ValueError(f"token ID {token_id} is outside vocabulary")
        flush()
        return "".join(pieces)
