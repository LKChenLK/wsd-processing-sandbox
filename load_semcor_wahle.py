import os
import argparse
import csv


from os import PathLike
from typing import Dict
from multiprocessing import Manager
from transformers import AutoTokenizer


INPUT_TEMPLATE = """\
question: which description describes the word " {0} " best in the \
following context? descriptions:[  " {1} ",  or " {2} " ] context: {3}\
"""

def load_keys(input_file: PathLike) -> None:
    data = []
    with open(input_file, "r", encoding="'iso-8859-1'") as f:
        reader = csv.reader(f, delimiter="\t")
        # Skip the header
        next(reader)        
        prev_target_id = None
        tmp_list = []
        for el in reader:
            if not prev_target_id:
                prev_target_id = el[0]
                continue
            if prev_target_id != el[0]:
                prev_target_id = el[0]
                data.append(tmp_list)
                tmp_list = []
            tmp_list.append(el)
        return data

def load_gold_keys(gold_key_file: PathLike) -> Dict[str, str]:
    gold_keys = {}
    with open(gold_key_file, "r", encoding="utf-8") as f:
        s = f.readline().strip()
        while s:
            tmp = s.split()
            gold_keys[tmp[0]] = tmp[1:]
            s = f.readline().strip()
    return gold_keys

def main(args):
    dataset_name = args.dataset_name
    data_dir = args.data_dir

    if 'semcor' not in dataset_name: # test data
        gold_keys = load_gold_keys(os.path.join(data_dir, 'gold_keys', f'{dataset_name}.gold.key.txt'))
        file_path = os.path.join(data_dir, 'examples', f'{dataset_name}_test_token_cls.csv')
        load_keys(os.path.join(data_dir, 'examples', f'{dataset_name}_test_token_cls.csv'))
    else: # training data
        file_path = os.path.join(data_dir, 'examples', f'{dataset_name}_train_token_cls.csv')
        load_keys(os.path.join(data_dir, 'examples', f'{dataset_name}_train_token_cls.csv'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(...)
    args = parser.parse_args()

    main(args)

