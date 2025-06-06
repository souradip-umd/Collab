from datasets import load_dataset
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from transfer_ma import ARGS
import time
import pickle
import torch
import numpy as np
import pdb
from random import shuffle
np.random.seed(42)
torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="Dahoas/full-hh-rlhf")
parser.add_argument("--split", type=str, default="train")

parser.add_argument("--run_percent", type=float, default=100.)
parser.add_argument("--rm_model", type=str)
parser.add_argument("--llm1", type=str)
parser.add_argument("--llm2", type=str)
parser.add_argument("--max_new_token", type=int, default=128)

parser.add_argument("--llm_gpu_1", type=str, default="cuda:0")
parser.add_argument("--llm_gpu_2", type=str, default="cuda:0")
parser.add_argument("--rm_gpu", type=str, default="cuda:1")
parser.add_argument("--recover", action='store_true', default = False)

parser.add_argument("--config", type=str)

parser.add_argument("--out_file", type=str)
parser.add_argument("--task_type", default = "direct", type=str)
args = parser.parse_args()

print(f"{args=}")

if args.recover:
    print("[INFO]: LOOKS LIKE YOU WANT TO RECOVER SOME RESULTS,")
    print("[INFO]: MAKE SURE ALL COMMANDLINE ARGS ARE EXACTLY THE SAME!!!")
    input("PRESS ENTER TO CONTINUE")

if not (args.max_new_token > 0):
    print("ERROR: Max tokens should be greater than 0!")
    exit(1)

cfg_path = Path(args.config)

if not cfg_path.exists():
    print("ERROR: Config doesn't exist!")
    exit(1)
    
out_path = Path(args.out_file + f"_0.jsonl")

if not out_path.exists() and args.recover:
    print("ERROR: out_path DOESN'T exist!")
    exit(1)

with open(cfg_path) as f:
    run_configs = [json.loads(line) for line in f.readlines()]
    
# validate configs
for run_config in run_configs:
    if "rm_weight" not in run_config:
        print(f"Missing key 'rm_weight' in {run_config=}")
        exit(1)
    elif "topk" not in run_config:
        print(f"Missing key 'topk' in {run_config=}")
        exit(1)
    elif "mode" not in run_config:
        print(f"Missing key 'mode' in {run_config=}")
        exit(1)
    elif "sample_temp" not in run_config:
        print(f"Missing key 'sample_temp' in {run_config=}")
        exit(1)

print(f"[INFO]: Loaded {len(run_configs)} run configs.")
print(f"[DEBUG]: {run_configs=}")
    
print(f"[INFO]: Loading dataset ({args.dataset=}, {args.split=})")
test_ds = load_dataset(args.dataset, split=args.split)


if args.dataset == "Dahoas/full-hh-rlhf":
    # FOR HHRLHF
    test_ds = test_ds["prompt"]

elif args.dataset == "openbmb/UltraFeedback":
    print("Running openbmb/UltraFeedback")
    test_ds = test_ds["instruction"]


elif args.dataset == "berkeley-nest/Nectar":
    print("Running berkeley-nest/Nectar")
    test_ds = test_ds["prompt"]


elif args.dataset == "openai/webgpt_comparisons":
    print("Running webgpt_comparisons")
    test_ds = test_ds["question"]
    


end_idx = int(len(test_ds) * (args.run_percent/100.))
print(f"[INFO]: {end_idx=}, {len(test_ds)=}")

truncated_ds = test_ds[0:end_idx]


print(f"{len(truncated_ds)=}")

print(f"[INFO]: Loading models ({args.llm=}, {args.rm=})")
search = ARGS(llm1=args.llm1, llm2 = args.llm2, rm=args.rm_model, llm_gpu_1=args.llm_gpu_1, llm_gpu_2=args.llm_gpu_2, rm_gpu=args.rm_gpu)
print(f"[INFO]: Done")

def runprompt(prompt: str, rm_weight=0., topk=5, new_token=24, mode="p_sigmoid_mixing", sample_temp=None, llm_dev:str="cuda:0") -> str:
    tokens, scores = search.generate(prompt, method=mode, topk=topk, max_new_token=new_token, weight=rm_weight, debug=False)

    # too long seqlen
    if tokens == None: return None, None
    
    raw_tokens = tokens[0].detach().cpu().numpy().tolist()
    tokens_text = search.tokens_to_text(tokens)[0]
    del tokens
    tokens_text_np = tokens_text.removeprefix(prompt)
    return tokens_text_np, raw_tokens, scores

for config_num, run_config in enumerate(run_configs):
    print(f"[INFO]: Running config: {run_config=}")

    data = []
    if args.recover and Path(args.out_file + f"_{config_num}.jsonl").exists():
        print(f"[INFO]: Run already exists, checking if it's done")
        resfile = open(Path(args.out_file + f"_{config_num}.jsonl"))
        samples = resfile.readlines()

        if samples[-1] != "":
            print("last line not empty??")
            exit(1)
        
        last_obj = json.loads(samples[-2])
        if last_obj["prompt"] != truncated_ds[len(samples) -1]:
            print(f"[INFO]: PROMPTS DID NOT MATCH RECOVERY FAILED!!!")
            exit(1)

    score_overall = []
    for idx, ds_row in enumerate(tqdm(truncated_ds)):
        if args.recover and (idx <= len(samples) -1):
            print(f"[INFO]: SKIPPING {idx}")
            continue

        print(f"{ds_row=}")
        current_prompt = ds_row 
        start = time.time()

        res, tokens, scores = runprompt(current_prompt, float(run_config["rm_weight"]), run_config["topk"], args.max_new_token, run_config["mode"], run_config["sample_temp"], llm_dev=args.llm_gpu)
        score_overall.append(scores)
        if tokens == None:
            print("Too long, skipped")
            continue

        elapsed = time.time() -start

        data.append({"prompt": current_prompt, "result": res, "response": current_prompt + res, "elapsed":elapsed, "method": args.out_file + f"_{config_num}"})
        print(f"[DEBUG]: {elapsed=} {len(current_prompt)=} {current_prompt=}, {res=}")
        with open(Path(args.out_file + f"_{config_num}.jsonl"), "w") as outfile:
            json.dump(data, outfile, ensure_ascii=False)


