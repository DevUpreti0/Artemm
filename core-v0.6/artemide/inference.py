"""Checkpoint-backed generation runtime. No external model provider is used."""
import torch
from artemide.model.config import ModelConfig
from artemide.model.transformer import ArtemideLM
from artemide.tokenizer import ByteTokenizer
from artemide.training.checkpoint import load_checkpoint

class InferenceRuntime:
    def __init__(self, checkpoint, config_path, device=None):
        self.device=device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config=ModelConfig.from_json(config_path)
        if self.config.vocab_size != ByteTokenizer.vocab_size:
            raise ValueError("Checkpoint config vocab_size does not match ByteTokenizer")
        self.model=ArtemideLM(self.config).to(self.device)
        state=load_checkpoint(checkpoint,self.model,map_location=self.device)
        self.model.eval(); self.step=int(state.get("step",0)); self.tokenizer=ByteTokenizer()
    @torch.inference_mode()
    def generate(self,prompt,max_new_tokens=128,temperature=0.8,top_k=40):
        ids=self.tokenizer.encode(prompt,add_bos=True)
        input_ids=torch.tensor([ids],dtype=torch.long,device=self.device)
        output=self.model.generate(input_ids,max_new_tokens=max_new_tokens,temperature=temperature,top_k=top_k)
        return self.tokenizer.decode(output[0].tolist()[len(ids):])
