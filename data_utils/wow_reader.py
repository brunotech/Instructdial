import csv
import logging
import json
import numpy as np
import os
import pickle

from collections import defaultdict
from tokenizers import BertWordPieceTokenizer
# from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Dict
import jsonlines

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

class WoWDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 # tokenizer: BertWordPieceTokenizer,
                 max_seq_length: int,
                 ):
        # For caching
        data_dirname = os.path.dirname(os.path.abspath(data_path))
        split = os.path.basename(os.path.abspath(data_path))
        lines = get_json_lines(f'{data_dirname}/{split}')
        self.examples = []
        self.examples.extend(iter(lines))
        self.idx=0

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def get_next(self):
        if self.idx>=len(self.examples):
            return None
        dp = self.examples[self.idx]
        self.idx+=1

        return dp
