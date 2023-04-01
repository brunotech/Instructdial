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

        dials = json.load(f_dials)

        turn_sys = ""
        turn_usr = ""

        for dial_dict in dials:
            dialog_history = []
            for ti, turn in enumerate(dial_dict["turns"]):
                if turn["speaker"] == "USER":
                    turn_usr = turn["utterance"].lower().strip()
                    data_detail = get_input_example("turn")
                    data_detail["ID"] = f"{ds_name}-{cnt_lin}"
                    data_detail["turn_id"] = ti % 2
                    data_detail["turn_usr"] = turn_usr
                    data_detail["turn_sys"] = turn_sys
                    data_detail["dialog_history"] = list(dialog_history)

                    if (not args["only_last_turn"]):
                        data.append(data_detail)

                    dialog_history.append(turn_sys)
                    dialog_history.append(turn_usr)

                elif turn["speaker"] == "SYSTEM":
                    turn_sys = turn["utterance"].lower().strip()

            if args["only_last_turn"]:
                data.append(data_detail)

            cnt_lin += 1
            if(max_line and cnt_lin >= max_line):
                break

    return data


def read_langs_dial(file_name, ontology, dialog_act, max_line = None, domain_act_flag=False):
    print(f"Reading from {file_name} for read_langs_dial")
    raise NotImplementedError



def prepare_data_schema(args):
    ds_name = "Schema"

    example_type = args["example_type"]
    max_line = args["max_line"]

    onlyfiles_trn = [
        os.path.join(
            args["data_path"], f'dstc8-schema-guided-dialogue/train/{f}'
        )
        for f in os.listdir(
            os.path.join(
                args["data_path"], "dstc8-schema-guided-dialogue/train/"
            )
        )
        if "dialogues" in f
    ]
    onlyfiles_dev = [
        os.path.join(
            args["data_path"], f'dstc8-schema-guided-dialogue/dev/{f}'
        )
        for f in os.listdir(
            os.path.join(
                args["data_path"], "dstc8-schema-guided-dialogue/dev/"
            )
        )
        if "dialogues" in f
    ]
    onlyfiles_tst = [
        os.path.join(
            args["data_path"], f'dstc8-schema-guided-dialogue/test/{f}'
        )
        for f in os.listdir(
            os.path.join(
                args["data_path"], "dstc8-schema-guided-dialogue/test/"
            )
        )
        if "dialogues" in f
    ]

    _example_type = "dial" if "dial" in example_type else example_type
    pair_trn = globals()[f"read_langs_{_example_type}"](
        args, onlyfiles_trn, max_line, ds_name
    )
    pair_dev = globals()[f"read_langs_{_example_type}"](
        args, onlyfiles_dev, max_line, ds_name
    )
    pair_tst = globals()[f"read_langs_{_example_type}"](
        args, onlyfiles_tst, max_line, ds_name
    )

    print(f"Read {len(pair_trn)} pairs train from {ds_name}")
    print(f"Read {len(pair_dev)} pairs valid from {ds_name}")
    print(f"Read {len(pair_tst)} pairs test  from {ds_name}")  

    meta_data = {"num_labels":0}

    return pair_trn, pair_dev, pair_tst, meta_data

