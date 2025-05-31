from typing import List
import torch
from torch.nn import functional as F
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, LlamaForCausalLM, LlamaForSequenceClassification
import pdb
import numpy as np
np.random.seed(42)
torch.manual_seed(42)
import sys

def factors(x):
    return [i for i in range(1,x+1) if x%i==0]

def auto_size(seq_len, topk):
    estimated = (28672/(seq_len*1.5)) -11.52605
    # hack
    possible_facs = factors(topk)
    if np.all(~(np.array(possible_facs[::-1]) < estimated)): return 1
    return possible_facs[::-1][np.argmax(np.array(possible_facs[::-1]) < estimated)]

###
def create_attention_mask(seq_len, bsz=1):
    return torch.ones((bsz, seq_len), device='cuda:0')

# From huggingface
def rcache(past_key_values, beam_idx):
    reordered_past = ()
    for layer_past in past_key_values:
        reordered_past += (
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
        )
    return reordered_past

def even_chunk(data, chunk_size=10):
    assert data.shape[0] % chunk_size == 0, "chunk_size must evenly divide the topk"
    for i in range(0, data.shape[0], chunk_size):
        yield data[i:(i+chunk_size)]

class ARGS:
    def __init__(self, llm1, llm2, rm, llm_gpu_1, llm_gpu_2, rm_gpu="cuda:1", torch_dtype=torch.float16):
       
        self.llm_gpu_1 = llm_gpu_1
        self.llm_gpu_2 = llm_gpu_2
        self.rm_gpu = rm_gpu
        print("Loading LLM...")
        self.LLM1 = AutoModelForCausalLM.from_pretrained(llm1, torch_dtype=torch_dtype).to(llm_gpu_1)
        self.LLM2 = AutoModelForCausalLM.from_pretrained(llm2, torch_dtype=torch_dtype).to(llm_gpu_2)
        self.LLM1.eval()
        self.LLM2.eval()
        self.tokenizer1 = AutoTokenizer.from_pretrained(llm1)
        self.tokenizer2 = AutoTokenizer.from_pretrained(llm2)
        print("Loading RM...")
        
        #target reward model
        self.RM = AutoModelForSequenceClassification.from_pretrained(rm, num_labels=1, torch_dtype=torch_dtype).to(rm_gpu)
        self.reward_tokenizer = AutoTokenizer.from_pretrained(rm)
        self.reward_tokenizer.pad_token = self.reward_tokenizer.eos_token
        self.RM.eval()
    
    def get_input_ids(self, prompt: str) -> torch.Tensor:
        tokens = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.llm_dev)
        return tokens
    
    def tokens_to_text(self, tokens: torch.Tensor) -> List[str]:
        return self.tokenizer1.batch_decode(tokens, skip_special_tokens=True)
    
    def generate_step(self, LLM, mout, input_ids, tokenizer, pre_screen_beam_width=40, weight=0., max_val=1, method="greedy", temperature=0.7, rm_cached=None, debug=True):
        out_logits = mout.logits[:, -1]
        prescreen_logits, prescreen_tokens = torch.topk(out_logits, dim=-1, k=pre_screen_beam_width)
        expanded_tis = torch.unsqueeze(input_ids, 1).repeat(1, pre_screen_beam_width, 1)
        
        if debug: print(f"{expanded_tis.shape=}")
        to_rm_eval = torch.dstack((expanded_tis, prescreen_tokens))
        if debug: print(f"{to_rm_eval.shape=}")
        if debug: print(f"{out_logits.shape[0] * pre_screen_beam_width=}")
        flat_trme = to_rm_eval.view(out_logits.shape[0] * pre_screen_beam_width, -1)
        if debug: print(f"{flat_trme.shape=}")
        
        
        #Traj specific
        flat_trme_ext = LLM.generate(flat_trme, max_new_tokens = max_val)
        output = [tokenizer.decode(r.squeeze()) for r in flat_trme_ext]

        texts_tokens = self.reward_tokenizer(output, return_tensors='pt', padding=True)
        for key, value in texts_tokens.items():
                 texts_tokens[key] = value.to(self.rm_gpu)
        outputs = self.RM(**texts_tokens)
        rm_out = outputs
        

        rewards = rm_out.logits.flatten().to(self.llm_gpu_1)
        
        del rm_out
        new_scores = rewards + prescreen_logits.flatten()
        
        
        if method == "greedy":
            _, top_k_ids = torch.topk(new_scores, dim=-1, k=1)
            top_reward, _ = torch.topk(new_scores, dim=-1, k=1)
        
        elif method == "topk":
            assert input_ids.shape[0] == 1
            new_scores = new_scores / temperature
            scores = F.softmax(new_scores, dim=-1)
            top_k_ids = torch.multinomial(scores, num_samples=1)
       
        else:
            raise ValueError(f"Invalid method '{method}'")
        
        
        return flat_trme[top_k_ids], new_scores[top_k_ids], 1
    
    def generate(self, prompt, weight=0., topk=1, max_new_token=128, method="greedy", temperature=0.7, chunk_size=5, debug=False):
        cached1 = None
        cached2 = None

        print(f"Max new tokens  = {max_new_token}")
        iterator_obj = range(max_new_token)
        if debug: iterator_obj = tqdm(iterator_obj)
        tokens = self.tokenizer1(prompt, return_tensors="pt").input_ids.to("cuda:0")
        tokens1 = tokens
        tokens2 = tokens

        rm_cached1 = None
        rm_cached2 = None
        max_abs =  128

        for i, _ in enumerate(iterator_obj):
            
            #hyper-prarmater
            max_val = 5

            #call llm1
            tokens1, rewards1, cached1, rm_cached1 = self.multiagent_generate(self.LLM1, self.tokenizer1, tokens, cached1, max_val, topk=topk, max_new_token=128, method="greedy", temperature=0.7, chunk_size=5, debug=False, rm_cached=rm_cached1)
            
            #call llm2
            tokens2, rewards2, cached2, rm_cached2 = self.multiagent_generate(self.LLM2, self.tokenizer2, tokens, cached2, max_val,  topk=topk, max_new_token=128, method="greedy", temperature=0.7, chunk_size=5, debug=False, rm_cached=rm_cached2)
            
            #Withou threholding
            thres = 0
            if rewards1.item() > rewards2.item() + thres  :
                tokens = tokens1
                
            else :
                tokens = tokens2
            
        
        return tokens, 1
    
    def multiagent_generate(self, LLM, tokenizer, tokens, cached, max_val, topk=1, max_new_token=128, method="greedy", temperature=0.7, chunk_size=5, debug=False, rm_cached=None):
        weight = 0.
        with torch.no_grad():
            if cached is None:
                mout = LLM(**LLM.prepare_inputs_for_generation(input_ids=tokens, attention_mask=create_attention_mask(tokens.shape[1], tokens.shape[0]).to("cuda:0"), past_key_values=None, use_cache=False))
                cached = mout.past_key_values
            else:
                mout = LLM(**LLM.prepare_inputs_for_generation(input_ids=tokens, attention_mask=create_attention_mask(tokens.shape[1], tokens.shape[0]).to("cuda:0"), past_key_values=cached, use_cache=True))
                cached = mout.past_key_values
            #output
            tokens, rewards, rm_cached = self.generate_step(LLM=LLM, mout= mout, input_ids = tokens, tokenizer = tokenizer, pre_screen_beam_width = topk, weight = weight, max_val = max_val, method = method, temperature = temperature, rm_cached = rm_cached, debug = debug)
            del mout
        return tokens, rewards, cached, rm_cached


