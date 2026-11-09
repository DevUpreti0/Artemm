"""Trainable byte-level BPE tokenizer for Artemide; no external tokenizer dependency."""
from __future__ import annotations
from collections import Counter
import heapq
import json
import os
from pathlib import Path
import tempfile

class BPETokenizer:
    format_name = "artemide-byte-bpe-v1"
    pad_token_id, bos_token_id, eos_token_id = 0, 1, 2
    def __init__(self, merges=()):
        self.merges = tuple((int(a), int(b)) for a,b in merges)
        self._merge_ids = {}
        self._token_bytes = [b"", b"", b""] + [bytes([i]) for i in range(256)]
        for index, pair in enumerate(self.merges):
            new_id = 259 + index
            left,right = pair
            if left < 3 or right < 3 or left >= new_id or right >= new_id:
                raise ValueError("BPE merges must reference earlier non-special token IDs")
            if pair in self._merge_ids:
                raise ValueError("duplicate BPE merge pair")
            self._merge_ids[pair] = new_id
            self._token_bytes.append(self._token_bytes[left] + self._token_bytes[right])
        self.vocab_size = len(self._token_bytes)

    @staticmethod
    def _merge_pair(sequence, pair, new_id):
        out=[]; i=0
        while i < len(sequence):
            if i+1 < len(sequence) and (sequence[i],sequence[i+1]) == pair:
                out.append(new_id); i += 2
            else:
                out.append(sequence[i]); i += 1
        return out

    @classmethod
    def train(cls, texts, *, vocab_size=1024, min_frequency=2):
        if not 259 <= vocab_size <= 65536:
            raise ValueError("vocab_size must be between 259 and 65536 for uint16 token storage")
        if min_frequency < 1:
            raise ValueError("min_frequency must be positive")
        sequences=[[byte+3 for byte in text.encode("utf-8")] for text in texts if isinstance(text,str) and text]
        if not sequences:
            raise ValueError("at least one non-empty training text is required")
        merges=[]
        while 259 + len(merges) < vocab_size:
            counts=Counter()
            for seq in sequences:
                counts.update(zip(seq,seq[1:]))
            if not counts:
                break
            pair,count=max(counts.items(),key=lambda item:(item[1],-item[0][0],-item[0][1]))
            if count < min_frequency:
                break
            new_id=259+len(merges)
            sequences=[cls._merge_pair(seq,pair,new_id) for seq in sequences]
            merges.append(pair)
        return cls(merges)

    def encode(self, text, *, add_bos=False, add_eos=False):
        if not isinstance(text,str):
            raise TypeError("text must be a string")
        ids=[byte+3 for byte in text.encode("utf-8")]
        if len(ids)>1 and self._merge_ids:
            n=len(ids); prev=[i-1 for i in range(n)]; nxt=[i+1 for i in range(n)]
            nxt[-1]=-1; alive=[True]*n; heap=[]
            def push(left,right):
                if left<0 or right<0: return
                rank=self._merge_ids.get((ids[left],ids[right]))
                if rank is not None: heapq.heappush(heap,(rank,left,right))
            for i in range(n-1): push(i,i+1)
            while heap:
                rank,left,right=heapq.heappop(heap)
                if not alive[left] or not alive[right] or nxt[left]!=right:
                    continue
                pair=(ids[left],ids[right])
                if self._merge_ids.get(pair)!=rank: continue
                ids[left]=rank; alive[right]=False
                after=nxt[right]; nxt[left]=after
                if after>=0: prev[after]=left
                before=prev[left]
                push(before,left); push(left,after)
            head=next((i for i in range(n) if alive[i] and prev[i]<0),-1)
            merged=[]
            while head>=0:
                merged.append(ids[head]); head=nxt[head]
            ids=merged
        if add_bos: ids.insert(0,self.bos_token_id)
        if add_eos: ids.append(self.eos_token_id)
        return ids

    def decode(self, token_ids, *, skip_special_tokens=True):
        output=bytearray(); pieces=[]
        def flush():
            if output:
                pieces.append(output.decode("utf-8",errors="replace")); output.clear()
        for raw in token_ids:
            token=int(raw)
            if 3<=token<self.vocab_size: output.extend(self._token_bytes[token])
            elif token in (0,1,2):
                if not skip_special_tokens:
                    flush(); pieces.append({0:"<pad>",1:"<bos>",2:"<eos>"}[token])
            else: raise ValueError(f"token ID {token} is outside vocabulary")
        flush()
        return "".join(pieces)

    def save(self,path):
        destination=Path(path); destination.parent.mkdir(parents=True,exist_ok=True)
        payload={"format":self.format_name,"vocab_size":self.vocab_size,"merges":[list(p) for p in self.merges]}
        fd,temp=tempfile.mkstemp(dir=destination.parent,prefix=destination.name+".",suffix=".tmp")
        try:
            with os.fdopen(fd,"w",encoding="utf-8") as handle:
                json.dump(payload,handle,ensure_ascii=False,indent=2); handle.write("\n")
            os.replace(temp,destination)
        finally:
            if os.path.exists(temp): os.unlink(temp)

    @classmethod
    def load(cls,path):
        with Path(path).open(encoding="utf-8") as handle: payload=json.load(handle)
        if payload.get("format")!=cls.format_name:
            raise ValueError("unsupported Artemide tokenizer format")
        tokenizer=cls(payload.get("merges",[]))
        if payload.get("vocab_size")!=tokenizer.vocab_size:
            raise ValueError("tokenizer vocabulary metadata does not match merges")
        return tokenizer
