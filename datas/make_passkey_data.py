import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

import argparse
import json
from utils.utils import set_seed
from utils.utils import get_promt
from utils.utils import get_tokenizer

import numpy as np
import os

TASK_DESCRIPT = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize it. I will quiz you about the important information there."

DEFAULT_CONTENT = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."

KEY_CONTENT = "The pass key is {KEY}. Remember it. {KEY} is the pass key."


class CreatePassKeyTask:
    def __init__(self, tokenizer):
        """
        data: List containing inputs
        """
        self.data = None
        self.tokenizer = tokenizer

        self.task_descript_len = len(self.tokenizer.encode(TASK_DESCRIPT)) + len(self.tokenizer.encode(get_promt("")))
        self.default_content_len = len(self.tokenizer.encode(DEFAULT_CONTENT))
        self.key_content_len = len(self.tokenizer.encode(KEY_CONTENT))

    def generate_passkey_data(self, max_token_length, num_examples=100, seed=0):

        num_distractors = (max_token_length - self.task_descript_len - self.key_content_len) // self.default_content_len

        rng = np.random.RandomState(seed)

        samples = []
        for _ in range(num_examples):
            random_answer = rng.randint(1,10000000)
            answer_sentence = KEY_CONTENT.format(KEY=random_answer)

            insert_location = rng.randint(0, num_distractors)
            input_ = [TASK_DESCRIPT] + [DEFAULT_CONTENT] * insert_location + [answer_sentence] + [DEFAULT_CONTENT] * (num_distractors - insert_location)

            text = " ".join(input_)
            samples.append({
                "text": text,
                "target": str(random_answer),
                "token_length": len(self.tokenizer.encode(text))
            })

        return samples



def main(args):

    tokenizer = get_tokenizer(args.model_path)
    task = CreatePassKeyTask(tokenizer)
    datas = []
    start_len, end_len, step = args.start_len, args.end_len, args.step
    max_token_length = start_len
    while max_token_length <= end_len:
        samples = task.generate_passkey_data(max_token_length, num_examples=args.repeat_times, seed=args.seed)
        datas.extend(samples)
        max_token_length += step

    with open(f"{os.path.join(os.getcwd(), args.dataset)}", "w", encoding="utf-8") as file:
        json.dump(datas, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="llama-7b")
    parser.add_argument("--start_len", type=int, default=1024)
    parser.add_argument("--end_len", type=int, default=16*1024)
    parser.add_argument("--step", type=int, default=1024)
    parser.add_argument("--dataset", type=str, default="passkey.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat_times", type=int, default=10)
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)