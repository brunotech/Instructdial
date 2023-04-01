import torch
import torch.utils.data as data
from .utils_function import to_cuda, merge, merge_multi_response, merge_sent_and_word

class Dataset_usdl(torch.utils.data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data_info, tokenizer, args, unified_meta, mode, max_length=512):
        """Reads source and target sequences from txt files."""
        self.data = data_info
        self.tokenizer = tokenizer
        self.num_total_seqs = len(data_info["ID"])
        self.usr_token = args["usr_token"]
        self.sys_token = args["sys_token"]
        self.usr_token_id = self.tokenizer.convert_tokens_to_ids(args["usr_token"])
        self.sys_token_id = self.tokenizer.convert_tokens_to_ids(args["sys_token"])
        self.max_length = max_length
        self.args = args
        self.unified_meta = unified_meta
        self.start_token = self.tokenizer.cls_token if "bert" in self.args["model_type"] else self.tokenizer.bos_token
        self.sep_token = self.tokenizer.sep_token if "bert" in self.args["model_type"] else self.tokenizer.eos_token
        self.mode = mode
        
    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        if self.args["example_type"] == "turn":
            dialog_history_str = self.get_concat_context(self.data["dialog_history"][index])
            context_plain = self.concat_dh_sys_usr(dialog_history_str, 
                                                   self.data["turn_sys"][index], 
                                                   self.data["turn_usr"][index])

            context = self.preprocess(context_plain)

        elif self.args["example_type"] == "dial":
            context_plain = self.data["dialog_history"][index]
            context = self.preprocess_slot(context_plain)

        return {
            "ID": self.data["ID"][index],
            "turn_id": self.data["turn_id"][index],
            "context": context,
            "context_plain": context_plain,
        }

    def __len__(self):
        return self.num_total_seqs
    
    def concat_dh_sys_usr(self, dialog_history, sys, usr):
        return f"{dialog_history} {self.sys_token} {sys}" + f" {self.usr_token} " + usr

    def preprocess(self, sequence):
        """Converts words to ids."""
        tokens = self.tokenizer.tokenize(self.start_token) + self.tokenizer.tokenize(sequence)[-self.max_length+1:]
        return torch.Tensor(self.tokenizer.convert_tokens_to_ids(tokens))

    def preprocess_slot(self, sequence):
        """Converts words to ids."""
        story = []
        for value in sequence:
            #v = list(self.tokenizer.encode(value))# + self.tokenizer.encode("[SEP]"))
            v = list(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value)))
            story.append(v)
        return story
    
    def get_concat_context(self, dialog_history):
        candidate_sys_responses = []
        dialog_history_str = "".join(
            f"{self.sys_token} {uttr} "
            if ui % 2 == 0
            else f"{self.usr_token} {uttr} "
            for ui, uttr in enumerate(dialog_history)
        )
        return dialog_history_str.strip()


def collate_fn_usdl_turn(data):
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x['context']), reverse=True) 

    item_info = {key: [d[key] for d in data] for key in data[0].keys()}
    # merge sequences
    src_seqs, src_lengths = merge(item_info['context'])

    item_info["context"] = to_cuda(src_seqs)
    item_info["context_len"] = src_lengths

    return item_info

def collate_fn_usdl_dial(data):
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x['context']), reverse=True) 

    item_info = {key: [d[key] for d in data] for key in data[0].keys()}
    # merge sequences
    src_seqs, src_lengths = merge_sent_and_word(item_info['context'])

    item_info["context"] = to_cuda(src_seqs)
    item_info["context_len"] = src_lengths

    return item_info

def collate_fn_usdl_dial_flat(data):
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x['context_flat']), reverse=True) 

    item_info = {key: [d[key] for d in data] for key in data[0].keys()}
    # merge sequences
    src_flat_seqs, src_flat_lengths = merge(item_info['context_flat'])
    src_seqs, src_lengths = merge_sent_and_word(item_info['context'])
    src_pos_seqs, src_pos_lengths = merge(item_info["sys_usr_id_positions"])

    item_info["context"] = to_cuda(src_seqs)
    item_info["context_len"] = src_lengths
    item_info["context_flat"] = to_cuda(src_flat_seqs)
    item_info["context_flat_len"] = src_flat_lengths
    item_info["sys_usr_id_positions"] = to_cuda(src_pos_seqs)

    return item_info

