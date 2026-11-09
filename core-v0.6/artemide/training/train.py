import argparse, json, os, time
import torch
from artemide.model.config import ModelConfig
from artemide.model.transformer import ArtemideLM
from artemide.training.checkpoint import save_checkpoint, load_checkpoint
from artemide.training.scheduler import lr_at_step
from artemide.training.telemetry import Telemetry

def load_tokens(path,device):
    data=torch.from_file(path,shared=False,size=os.path.getsize(path)//2,dtype=torch.uint16).to(torch.long)
    if data.numel()<2: raise ValueError(f"Dataset {path} has fewer than two tokens")
    return data.to(device)

def get_batch(tokens,batch_size,seq_len):
    if tokens.numel()<=seq_len: raise ValueError(f"Dataset needs more than seq_len={seq_len} tokens; found {tokens.numel()}")
    starts=torch.randint(0,tokens.numel()-seq_len-1,(batch_size,),device=tokens.device)
    x=torch.stack([tokens[i:i+seq_len] for i in starts])
    y=torch.stack([tokens[i+1:i+seq_len+1] for i in starts])
    return x,y

@torch.no_grad()
def evaluate(model,tokens,batch_size,seq_len,batches=10):
    model.eval(); losses=[]
    for _ in range(batches):
        x,y=get_batch(tokens,batch_size,seq_len); _,loss=model(x,y); losses.append(float(loss))
    model.train(); return sum(losses)/len(losses)

def main():
    p=argparse.ArgumentParser(description="Train Artemide from a prepared real-text corpus")
    p.add_argument("--config",required=True); p.add_argument("--data",required=True)
    p.add_argument("--steps",type=int,default=1000); p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--seq-len",type=int,default=128); p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--warmup",type=int,default=50); p.add_argument("--min-lr",type=float,default=3e-5)
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out",default="runs"); p.add_argument("--resume",default=None)
    p.add_argument("--eval-interval",type=int,default=100); p.add_argument("--eval-batches",type=int,default=10)
    a=p.parse_args()
    if a.steps<1 or a.eval_interval<1: p.error("steps and eval-interval must be positive")
    cfg=ModelConfig.from_json(a.config)
    if cfg.vocab_size!=259: p.error("ByteTokenizer requires vocab_size=259")
    train_tokens=load_tokens(os.path.join(a.data,"train.bin"),a.device)
    val_path=os.path.join(a.data,"validation.bin")
    val_tokens=load_tokens(val_path,a.device) if os.path.exists(val_path) else None
    model=ArtemideLM(cfg).to(a.device); opt=torch.optim.AdamW(model.parameters(),lr=a.lr); start=0
    if a.resume:
        state=load_checkpoint(a.resume,model,opt,map_location=a.device); start=int(state.get("step",0))
    os.makedirs(a.out,exist_ok=True); telemetry=Telemetry(os.path.join(a.out,"metrics.jsonl"))
    model.train(); began=time.time()
    for step in range(start,start+a.steps):
        lr=lr_at_step(step,a.warmup,start+a.steps,a.lr,a.min_lr)
        for group in opt.param_groups: group["lr"]=lr
        x,y=get_batch(train_tokens,a.batch_size,a.seq_len)
        _,loss=model(x,y); opt.zero_grad(set_to_none=True); loss.backward()
        grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        row={"step":step+1,"train_loss":float(loss.detach()),"learning_rate":lr,
             "grad_norm":float(grad_norm),"tokens_seen":(step-start+1)*a.batch_size*a.seq_len,
             "tokens_per_second":int((step-start+1)*a.batch_size*a.seq_len/max(time.time()-began,1e-6))}
        if val_tokens is not None and (step+1)%a.eval_interval==0:
            row["validation_loss"]=evaluate(model,val_tokens,a.batch_size,a.seq_len,a.eval_batches)
        telemetry.log(**row)
        if (step+1)%a.eval_interval==0:
            save_checkpoint(os.path.join(a.out,"latest.pt"),model,opt,step+1,extra={"config":cfg.__dict__})
            print(json.dumps(row),flush=True)
    save_checkpoint(os.path.join(a.out,"final.pt"),model,opt,start+a.steps,extra={"config":cfg.__dict__})
    print(f"Training complete: {start+a.steps} total steps. Checkpoint: {a.out}/final.pt")
if __name__=="__main__": main()
