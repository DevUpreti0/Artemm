"""Command-line entry point for Artemide Core."""
import argparse, json
from pathlib import Path
import torch
from .data import prepare_corpus, TokenStream, load_documents
from .inference import load_model, generate_text
from .model.config import ModelConfig
from .model.transformer import ArtemideLM
from .trainer import run_training
from .training import save_checkpoint
from .tokenizer import ByteTokenizer
from .bpe_tokenizer import BPETokenizer

def main():
    parser=argparse.ArgumentParser(prog="artemide")
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("prepare",help="tokenize a licensed text corpus")
    p.add_argument("--input",required=True); p.add_argument("--output",default="data/processed")
    p.add_argument("--validation-fraction",type=float,default=0.05); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--license",default="unspecified",dest="license_name")
    p.add_argument("--tokenizer-file")
    t=sub.add_parser("train",help="train from uint16 token files")
    t.add_argument("--config",required=True); t.add_argument("--train",required=True); t.add_argument("--validation")
    t.add_argument("--output",default="runs/artemide"); t.add_argument("--steps",type=int,default=1000)
    t.add_argument("--batch-size",type=int,default=8); t.add_argument("--sequence-length",type=int,default=256)
    t.add_argument("--learning-rate",type=float,default=3e-4); t.add_argument("--weight-decay",type=float,default=0.1)
    t.add_argument("--warmup-steps",type=int,default=100); t.add_argument("--grad-accum-steps",type=int,default=1)
    t.add_argument("--precision",choices=["auto","fp32","bf16","fp16"],default="auto"); t.add_argument("--resume")
    t.add_argument("--validation-interval",type=int,default=100); t.add_argument("--checkpoint-interval",type=int,default=500)
    t.add_argument("--seed",type=int,default=42)
    g=sub.add_parser("generate",help="generate from a trusted checkpoint")
    g.add_argument("--config",required=True); g.add_argument("--checkpoint",required=True); g.add_argument("--prompt",required=True)
    g.add_argument("--max-new-tokens",type=int,default=128); g.add_argument("--temperature",type=float,default=0.0)
    g.add_argument("--top-k",type=int,default=40); g.add_argument("--tokenizer-file")
    b=sub.add_parser("bpe-train",help="train a byte-level BPE tokenizer from permitted text")
    b.add_argument("--input",required=True); b.add_argument("--output",required=True)
    b.add_argument("--vocab-size",type=int,default=1024); b.add_argument("--min-frequency",type=int,default=2)
    i=sub.add_parser("init",help="create an untrained random-initialization checkpoint")
    i.add_argument("--config",required=True); i.add_argument("--output",default="runs/artemide/initial.pt")
    i.add_argument("--seed",type=int,default=42)
    s=sub.add_parser("serve",help="start the optional HTTP API")
    s.add_argument("--host",default="0.0.0.0"); s.add_argument("--port",type=int,default=8000)
    args=parser.parse_args()
    if args.command=="bpe-train":
        docs=load_documents(args.input)
        tokenizer=BPETokenizer.train([text for _,text in docs],vocab_size=args.vocab_size,min_frequency=args.min_frequency)
        tokenizer.save(args.output)
        print(json.dumps({"tokenizer":args.output,"vocab_size":tokenizer.vocab_size,"merges":len(tokenizer.merges)},indent=2)); return
    if args.command=="init":
        torch.manual_seed(args.seed)
        with Path(args.config).open(encoding="utf-8") as f: config=ModelConfig(**json.load(f))
        model=ArtemideLM(config)
        save_checkpoint(args.output,model,step=0,metadata={"stage":"random-initialization","seed":args.seed})
        print(json.dumps({"checkpoint":args.output,"step":0,"stage":"random-initialization"},indent=2))
        return
    if args.command=="prepare":
        print(json.dumps(prepare_corpus(args.input,args.output,validation_fraction=args.validation_fraction,
            seed=args.seed,license_name=args.license_name,
            tokenizer=BPETokenizer.load(args.tokenizer_file) if args.tokenizer_file else ByteTokenizer()),indent=2)); return
    if args.command=="train":
        with Path(args.config).open(encoding="utf-8") as f: config=ModelConfig(**json.load(f))
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model=ArtemideLM(config).to(device); train_data=TokenStream(args.train,config.vocab_size)
        val_data=TokenStream(args.validation,config.vocab_size) if args.validation else None
        print(json.dumps(run_training(model,train_data,output_dir=args.output,steps=args.steps,batch_size=args.batch_size,
            sequence_length=args.sequence_length,learning_rate=args.learning_rate,weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,grad_accum_steps=args.grad_accum_steps,precision=args.precision,
            validation_data=val_data,validation_interval=args.validation_interval,
            checkpoint_interval=args.checkpoint_interval,resume=args.resume,seed=args.seed),indent=2)); return
    if args.command=="generate":
        model,tokenizer,device=load_model(args.config,args.checkpoint,tokenizer_path=args.tokenizer_file)
        print(generate_text(model,tokenizer,args.prompt,device=device,max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,top_k=args.top_k)); return
    if args.command=="serve":
        try: import uvicorn
        except ImportError as exc: raise SystemExit("Install API extras: pip install 'artemide-core[api]'") from exc
        from .api import create_app
        uvicorn.run(create_app(),host=args.host,port=args.port)

if __name__=="__main__":
    main()
