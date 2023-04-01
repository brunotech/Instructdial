import json
import ast
import collections
import os

from .utils_function import get_input_example


def read_langs_turn(args, file_name, max_line = None, ds_name=""):
    print(f"Reading from {file_name} for read_langs_turn")

    data = []

    with open(file_name) as f:
        dials = json.load(f)

        cnt_lin = 1
        for dial_dict in dials:
            dialog_history = []

            turn_usr = ""
            turn_sys = ""
            for ti, turn in enumerate(dial_dict["dialogue"]):
                if turn["turn"] == "driver":
                    turn_usr = turn["data"]["utterance"].lower().strip()

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
                elif turn["turn"] == "assistant":
                    turn_sys = turn["data"]["utterance"].lower().strip()

            if args["only_last_turn"]:
                data.append(data_detail)

            cnt_lin += 1
            if(max_line and cnt_lin >= max_line):
                break

    return data


def read_langs_dial(file_name, ontology, dialog_act, max_line = None, domain_act_flag=False):
    print(f"Reading from {file_name} for read_langs_dial")
    raise NotImplementedError



def prepare_data_smd(args):
    ds_name = "SMD"

    example_type = args["example_type"]
    max_line = args["max_line"]

    file_trn = os.path.join(args["data_path"], "kvret/kvret_train_public.json")
    file_dev = os.path.join(args["data_path"], "kvret/kvret_dev_public.json")
    file_tst = os.path.join(args["data_path"], "kvret/kvret_test_public.json")

    _example_type = "dial" if "dial" in example_type else example_type
    pair_trn = globals()[f"read_langs_{_example_type}"](
        args, file_trn, max_line, ds_name
    )
    pair_dev = globals()[f"read_langs_{_example_type}"](
        args, file_dev, max_line, ds_name
    )
    pair_tst = globals()[f"read_langs_{_example_type}"](
        args, file_tst, max_line, ds_name
    )

    print(f"Read {len(pair_trn)} pairs train from {ds_name}")
    print(f"Read {len(pair_dev)} pairs valid from {ds_name}")
    print(f"Read {len(pair_tst)} pairs test  from {ds_name}")  

    meta_data = {"num_labels":0}

    return pair_trn, pair_dev, pair_tst, meta_data

