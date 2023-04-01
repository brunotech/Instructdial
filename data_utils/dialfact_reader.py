import csv
import logging
import json
import numpy as np
import os
import pickle
import string
import random
from collections import defaultdict
from tokenizers import BertWordPieceTokenizer
# from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Dict
import jsonlines
import json
from constants import SPECIAL_TOKENS

from data_utils.data_reader import Dataset

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

def get_json_lines(inp_file):
    lines = []
    with jsonlines.open(inp_file) as reader:
        lines.extend(iter(reader))
    return lines

# def get_json_lines(inp_file):
#     lines = []
#     with open(inp_file, 'r') as reader:
#         for obj in reader:
#             import pdb;pdb.set_trace()
#             lines.append(obj)
#     return lines

class DialfactDataset(Dataset):
    def __init__(self,split='train'):
        self.split = split
        # For caching
        # data_dirname = os.path.dirname(os.path.abspath(data_path))
        # split = os.path.basename(os.path.abspath(data_path))
        self.examples = []
        self.labels = ['supports', 'refutes', 'not enough info']
        self.name = 'dnli'

        if split in ['train', 'dev']:
            data = get_json_lines('./datasets/dialfact/valid_split.jsonl')
            if split == 'train':
                data = data[:int(len(data)//2)]
            elif split == 'dev':
                data = data[int(len(data)//2):]
        if split == 'test':
            data = get_json_lines('./datasets/dialfact/test_split.jsonl')

        for odp in tqdm(data):
            # print(odp)
            # import pdb;pdb.set_trace()
            dialogue_context = list(odp['context'])
            dp = {
                'context': dialogue_context,
                'response': odp['response'],
                'label': odp['response_label'].lower(),
            }
            dp |= odp
            self.examples.append(dp)


        print('data length', len(self.examples))

        # for i, dialogue in enumerate(lines):
        #     self.examples.append(dialogue)

        self.idx=0

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


