import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

import argparse
import pickle
from collections import defaultdict

import torch.types
from tqdm import tqdm

from datas.get_data import get_data
from torch.utils.data import DataLoader
from utils.utils import get_model, get_promt, compare_retrieval_acc, set_seed

import numpy as np
import os

model_custom_config = {
    "max_new_tokens": 50,
    "temperature": 0.1,
    "top_p": 0.9
}

def main(args):

    dataset = get_data(args.dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    if args.cuda == "auto":
        device = "auto"
    else:
        device = torch.device(int(args.cuda))

    tokenizer, model = get_model(args.model_path, device, method=args.method, args=args)

    prefix_prompt = get_promt(args.model_path)

    all_length_acc = defaultdict(list)

    pbar = tqdm(dataloader)
    for data in pbar:
        model.eval()
        with torch.no_grad():
            query = prefix_prompt.format(data["text"][0])
            inputs_token = tokenizer(query, return_tensors="pt").to(model.device)
            input_ids = inputs_token.input_ids
            print("input token length: {}".format(len(input_ids[0])))
            #
            if len(input_ids[0]) < 10000:
                continue

            outputs = model.generate(input_ids, **model_custom_config)

            response = tokenizer.decode(outputs[0])[len(query):]
            print("response: {}".format(response))
            print("target: {}".format(data["target"][0]))

            acc = compare_retrieval_acc(response, data["target"][0])

            if acc == 1:
                print("success")
            else:
                print("failed")

            token_length = int(data["token_length"][0])
            all_length_acc[token_length].append(acc)

        all_mean_var_res = {
            token_length: {
                "mean": np.nanmean(np.array(record)),
                "var": np.nanvar(np.array(record))
            }
            for token_length, record in all_length_acc.items()
        }

        with open(f"{os.path.join(os.getcwd(), args.log_dir)}/{args.save_file}", "wb") as f:
            pickle.dump({"length_mean_var": all_mean_var_res}, f)

        with open(f"{os.path.join(os.getcwd(), args.log_dir)}/{'record_'+args.save_file}", "wb") as f:
            pickle.dump({"all_length_acc": all_length_acc}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/data/persist/models/llama/llama2/llama2-7b-chat")  # mosaicml-mpt-7b
    parser.add_argument("--method", type=str, default="mesa-extrapolation")
    parser.add_argument("--dataset", type=str, default="../datas/passkey-data_dup-10_answer-6bit.json")
    parser.add_argument("--save_file", type=str, default="_test.pkl")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--log_dir", type=str, default="../logs")
    parser.add_argument("--cuda", type=str, default="auto")
    parser.add_argument("--hard_cuda", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    # for mpt-alibi
    parser.add_argument("--push_mpt", type=int, default=512)
    parser.add_argument("--push_width", type=int, default=50)
    parser.add_argument("--chunk_width", type=int, default=512)

    args = parser.parse_args()
    set_seed(args.seed)
    main(args)

















