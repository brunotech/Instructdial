import json
import ast
import collections
import os

from .utils_function import get_input_example


def read_langs_turn(args, dial_files, max_line = None, ds_name=""):
    print(f"Reading from {ds_name} for read_langs_turn")

    data = []

    cnt_lin = 1
    for dial_file in dial_files:
        
        f_dials = open(dial_file, 'r')
        dials = f_dials.readlines()

        for dial in dials:
            dialog_history = []
            dial_dict = json.loads(dial)
            # Reading data
            for ti, turn in enumerate(dial_dict["turns"]):
                if ti%2 == 0:
                    turn_sys = turn.lower().strip()
                else:
                    turn_usr = turn.lower().strip()
                    data_detail = get_input_example("turn")
                    data_detail["ID"] = f"{ds_name}-{cnt_lin}"
                    data_detail["turn_id"] = ti % 2
                    data_detail["turn_usr"] = turn_usr
                    data_detail["turn_sys"] = turn_sys
                    data_detail["dialog_history"] = list(dialog_history)

                    if not args["only_last_turn"]:
                        data.append(data_detail)

                    dialog_history.append(turn_sys)
                    dialog_history.append(turn_usr)

            if args["only_last_turn"]:
                data.append(data_detail)

            cnt_lin += 1
            if(max_line and cnt_lin >= max_line):
                break

    return data


def read_langs_dial(file_name, ontology, dialog_act, max_line = None, domain_act_flag=False):
    print(f"Reading from {file_name} for read_langs_dial")

    raise NotImplementedError



def prepare_data_metalwoz(args):
    ds_name = "MetaLWOZ"

    example_type = args["example_type"]
    max_line = args["max_line"]

    onlyfiles = [
        os.path.join(args["data_path"], f'metalwoz/dialogues/{f}')
        for f in os.listdir(
            os.path.join(args["data_path"], "metalwoz/dialogues/")
        )
        if ".txt" in f
    ]

    _example_type = "dial" if "dial" in example_type else example_type
    pair_trn = globals()[f"read_langs_{_example_type}"](
        args, onlyfiles, max_line, ds_name
    )
    pair_dev = []
    pair_tst = []

    print(f"Read {len(pair_trn)} pairs train from {ds_name}")
    print(f"Read {len(pair_dev)} pairs valid from {ds_name}")
    print(f"Read {len(pair_tst)} pairs test  from {ds_name}")  

    meta_data = {"num_labels":0}

    return pair_trn, pair_dev, pair_tst, meta_data

