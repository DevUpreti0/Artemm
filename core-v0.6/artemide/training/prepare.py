import argparse, json
from artemide.training.corpus import prepare_corpus

def main():
    p=argparse.ArgumentParser(description="Prepare licensed Artemide text data")
    p.add_argument("--input",required=True); p.add_argument("--output",default="data/processed")
    p.add_argument("--validation-fraction",type=float,default=0.05); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--license",default="unspecified")
    a=p.parse_args()
    if not 0 <= a.validation_fraction < 1: p.error("validation-fraction must be in [0,1)")
    print(json.dumps(prepare_corpus(a.input,a.output,a.validation_fraction,a.seed,a.license),indent=2))
if __name__=="__main__": main()
