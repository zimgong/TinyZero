"""
Preprocess dataset for sudoku task - given a sudoku board, generate the solution
"""

import re
import os
from datasets import Dataset, load_dataset
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse


def make_prefix(dp, template_type):
    puzzle = dp['puzzle']
    puzzle_formatted = "["
    for i in range(9):
        line = "[" + ",".join(puzzle[i*9:(i+1)*9].replace("0", "*")) + "]"
        puzzle_formatted += line + ","
    puzzle_formatted = puzzle_formatted[:-1] + "]"
    # NOTE: also need to change reward_score/sudoku.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Solve this 9x9 sudoku puzzle {puzzle_formatted} where * represents a cell to be filled in. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags in the following format: <answer> [[*,*,*,*,*,*,*,*,*],[*,*,*,*,*,*,*,*,*],[*,*,*,*,*,*,*,*,*],[*,*,*,*,*,*,*,*,*],[*,*,*,*,*,*,*,*,*],[*,*,*,*,*,*,*,*,*],[*,*,*,*,*,*,*,*,*],[*,*,*,*,*,*,*,*,*],[*,*,*,*,*,*,*,*,*]] </answer>.
Assistant: Let me solve this step by step.
<think>"""
    elif template_type == 'qwen-instruct':
        """This works for Qwen Instruct Models"""
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n Solve this 9x9 sudoku puzzle {puzzle_formatted} where * represents a cell to be filled in. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags in the following format: <answer> [[*,*,*,*,*,*,*,*,*],[*,*,*,*,*,*,*,*,*],[*,*,*,*,*,*,*,*,*],[*,*,*,*,*,*,*,*,*],[*,*,*,*,*,*,*,*,*],[*,*,*,*,*,*,*,*,*],[*,*,*,*,*,*,*,*,*],[*,*,*,*,*,*,*,*,*],[*,*,*,*,*,*,*,*,*]] </answer>.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='dataset/sudoku')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--num_samples', type=int, default=100000)
    parser.add_argument('--num_operands', type=int, default=6)
    parser.add_argument('--max_target', type=int, default=1000)
    parser.add_argument('--min_number', type=int, default=1)
    parser.add_argument('--max_number', type=int, default=100)
    parser.add_argument('--train_size', type=int, default=327680)
    parser.add_argument('--test_size', type=int, default=1024)
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    data_source = 'sudoku'
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    raw_dataset = load_dataset('Ritvik19/Sudoku-Dataset')

    # assert len(raw_dataset) > TRAIN_SIZE + TEST_SIZE
    train_dataset = raw_dataset['train'].select(range(TRAIN_SIZE))
    test_dataset = raw_dataset['validation'].select(range(TEST_SIZE))

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, template_type=args.template_type)
            solution = {
                "puzzle": example['puzzle'],
                "solution": example['solution']
            }
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('validation'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir) 
