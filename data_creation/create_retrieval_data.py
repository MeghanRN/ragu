from tqdm import tqdm
import jsonlines
import argparse
import json
import os
import csv
from read_dataset import get_entry_from_dataset
import sys
sys.path.append('../utils')
from utils import load_jsonlines, save_file_jsonl, load_csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file')
    parser.add_argument('--dataset')
    parser.add_argument('--output_file', type=str)
    args = parser.parse_args()
    processed_data = []

    if args.input_file.endswith(".json"):
        data = json.load(open(args.input_file))
    elif args.input_file.endswith(".jsonl"):
        data = load_jsonlines(args.input_file)
    elif args.input_file.endswith(".csv") or args.input_file.endswith(".tsv"):
        if args.dataset in ['WebQ', 'SQuAD', 'PopQA']:
            data = load_csv(args.input_file)

    if args.dataset in ['NQ', 'TQA']:
        data = data['data']

    for idx, item in tqdm(enumerate(data)):
        processed_data.append(get_entry_from_dataset(args.dataset, item, idx))

    print(len(processed_data))
    save_file_jsonl(processed_data, args.output_file)

if __name__ == "__main__":
    main()
