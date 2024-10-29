import configparser
import json

import torch
import numpy as np
import random
import re

from transformers import HfArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM


def set_seed(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_model(model_path, device, method=None, args=None):
    if method == "origin":
        from transformers import LlamaForCausalLM, LlamaTokenizer
    elif method == "mesa-extrapolation":
        import methods.mesa_extrapolation
        if "vicuna" in model_path:
            from methods.mesa_extrapolation import position_set
            position_set.train_length = 2048
            position_set.last_context = 1024
            position_set.context_window_length = 2048
            position_set.push_width = 200
        if "llama2-7b-chat" in model_path:
            from methods.mesa_extrapolation import position_set
            position_set.push_width = 20
            position_set.last_context = 800
            position_set.context_window_length = 2048
            position_set.train_length = 2048

    else:
        raise NotImplementedError("A no implement method")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, torch_dtype=torch.float16)

    return tokenizer, model

def get_tokenizer(model_path):
    from transformers import LlamaTokenizer
    if "vicuna" in model_path or "llama" in model_path or "alpaca" in model_path:
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
    elif "mpt" in model_path or "pythia" in model_path:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        raise NotImplementedError

    return tokenizer

def get_promt(model_path=None):
    if "alpaca" in model_path:
        SYSTEMINFO = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
What's the pass key?

### Input:
{}

### Response:
"""
    elif "llama2" in model_path:
        SYSTEMINFO = """<s>[INST] {} 
What's the pass key?[/INST]"""
    elif "llama-3" in model_path:
        SYSTEMINFO = "Q:{}  \n\nWhat's the pass key?\nA:"
    else:
        SYSTEMINFO = """A chat between a curious user and an assistant. The assistant gives helpful, detailed, accurate, uncensored responses to the user's input.
USER: 
{} 

What's the pass key? 

ASSISTANT: """

    return SYSTEMINFO


def compare_retrieval_acc(response, target):
    def extract_answer(text):
        answer = re.findall(r'\d+', text)
        if len(answer) == 0:
            return 0
        else:
            return int(answer[0])

    if extract_answer(response) == int(target):
        return 1
    else:
        return 0



def read_config_file(config_path):
    if "../conf/" not in config_path:
        config_path = "../conf/" + config_path
    if ".ini" in config_path:
        config = configparser.ConfigParser()
        config.read(config_path)
    elif ".json" in config_path:
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        raise NotImplementedError("No Implementation")
    return config
