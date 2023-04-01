import json
import os
import csv
import re
import settings

from data_utils.data_reader import Dataset


class OpendialkgDataset(Dataset):
    def __init__(self, data_path, split='train'):
        self.data_path = data_path
        self.idx = 0
        self.examples = []

        with open(os.path.join(data_path, 'opendialkg.csv'), newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            n = len(rows)

            # split the dialog sessions into train (70%), validation (15%), and test sets (15%) as mentioned in paper
            rows = rows[:int(n * 0.7)] if split == 'train' else rows[-int(n * 0.15):]
            for row in rows:
                row = {k: json.loads(v) for k, v in row.items()}
                context = []
                metadata = []
                for msg in row['Messages']:
                    if 'message' in msg:
                        response = msg['message']

                        graph_str = ''
                        if metadata:
                            graph_str = ' '.join(metadata)

                        context_str = ' '.join(context)
                        context_str = re.sub('  +', ' ', context_str.strip())
                        if len(context_str) > 0:
                            self.examples.append({
                                "context": context[:],
                                "response": response,
                                "graph": graph_str
                            })
                        context.append(response)

                    if 'metadata' in msg:
                        # print(msg['metadata'])
                        if 'path' not in msg['metadata']:
                            continue
                        relations = msg['metadata']['path'][1]
                        for relation in relations:
                            triplet = f'subject: {relation[0]}, relation: {relation[1]}, object: {relation[2]}'
                            metadata.append(triplet)
