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

def longestSubstringFinder(S,T):
    m = len(S)
    n = len(T)
    counter = [[0]*(n+1) for _ in range(m+1)]
    longest = 0
    lcs_set = set()
    for i in range(m):
        for j in range(n):
            if S[i] == T[j]:
                c = counter[i][j] + 1
                counter[i+1][j+1] = c
                if c > longest:
                    longest = c
                    lcs_set = {S[i - longest + 1:i+1]}
                elif c == longest:
                    lcs_set.add(S[i-c+1:i+1])

    return lcs_set

def clean_target_fromresponse(response, target):
    targetnopunc = target.translate(str.maketrans('', '', string.punctuation)).lower()
    response = response.replace(target, '').replace(target.lower(), '').replace(targetnopunc, '').strip()

    return response

class OttersDataset(Dataset):
    def __init__(self,split='train',domain='both'):
        self.split = split
        # For caching
        # data_dirname = os.path.dirname(os.path.abspath(data_path))
        # split = os.path.basename(os.path.abspath(data_path))
        self.examples = []

        if domain in ['in_domain', 'both']:
            folder = './datasets/OTTers/data/in_domain/'
            with open(folder + split +'/source.csv') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                all_data = list(csv_reader)
                        # print(f'Processed {line_count} lines.')
            with open(f'{folder}/' + split + '/target.csv') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                all_target = list(csv_reader)
                        # print(f'Processed {line_count} lines.')

            combined_data = []
            for i in range(len(all_data)):
                source = all_data[i][1]
                target = all_data[i][2]
                target_clause = target
                response = all_target[i][1]
                d = {
                    'target': target_clause,
                    'response': clean_target_fromresponse(response, target),
                }
                d['responsewithtarget']= d['response']+' ' +target
                d['context'] = source
                d['domain'] = 'in_domain'
                combined_data.append(d)

            print('indomain data length', len(combined_data))
            self.examples+=combined_data

        if domain in ['out_of_domain', 'both']:
            folder = './datasets/OTTers/data/out_of_domain/'
            with open(folder + split +'/source.csv') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                all_data = list(csv_reader)
                        # print(f'Processed {line_count} lines.')
            with open(folder +'/' + split +'/target.csv') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                all_target = list(csv_reader)
                        # print(f'Processed {line_count} lines.')

            combined_dataout = []
            for i in range(len(all_data)):
                source = all_data[i][1]
                target = all_data[i][2]
                target_clause = target
                response = all_target[i][1]
                d = {}
                d['target'] = target_clause
                d['response'] = clean_target_fromresponse(response, target)
                # d['responsewithtarget'] = []
                # matched = longestSubstringFinder(response.lower(), target.lower())
                # matched = list(matched)[0]
                # if len(matched)>len(target)-5:
                #     d['responsewithtarget'].append(d['response'])
                # else:
                #     d['responsewithtarget'].append(d['response']+' ' +target)
                # d['responsewithtarget'] = d['responsewithtarget'][0]
                d['responsewithtarget']= d['response']+' ' +target
                d['context'] = source
                d['domain'] = 'out_of_domain'
                combined_dataout.append(d)

            self.examples+=combined_dataout
            print('outofdomain data length', len(combined_dataout))

        print('data length', len(self.examples))

        # for i, dialogue in enumerate(lines):
        #     self.examples.append(dialogue)

        self.idx=0

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


