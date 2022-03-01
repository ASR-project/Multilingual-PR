import torch
from torch.nn.utils.rnn import pad_sequence
import os
import os.path as osp
import json

import re
from datasets import load_dataset
from utils.logger import init_logger


def coll_fn(batch):
    batch_dict={}
    list_tensors = [torch.from_numpy(b['audio']['array']) for b in batch] 
    batch_dict['array'] = pad_sequence(list_tensors, padding_value=-100, batch_first=True)
    batch_dict['path'] = [b['path'] for b in batch]
    batch_dict['sentence'] = [b['sentence'] for b in batch]

    return batch_dict

def create_vocabulary(data_param):
    
    logger = init_logger("create_vocabulary", "INFO")

    vocab_path = osp.join(os.getcwd(), "assets", "vocab")
    file_dict = os.path.join(vocab_path, f"vocab-{data_param.dataset_name}-{data_param.subset}.json")

    if os.path.isfile(file_dict):
        logger.info(f'{file_dict} already exists')
        return file_dict

    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

    def remove_special_characters(batch):
        batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
        return batch

    def extract_all_chars(batch):
        all_text = " ".join(batch["sentence"])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    dataset = load_dataset(data_param.dataset_name, data_param.subset,
                            use_auth_token=data_param.use_auth_token, 
                            download_mode=data_param.download_mode, 
                            cache_dir=data_param.cache_dir)

    dataset = dataset.map(remove_special_characters)
    vocabs = dataset.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=dataset.column_names["train"])

    vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))

    vocab_dict = {v: k for k, v in enumerate(vocab_list)}

    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]

    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    logger.info(f'Length vocabulary : {vocab_dict}')

    if not os.path.exists(vocab_path):
        os.makedirs(vocab_path)

    with open(file_dict, "w") as vocab_file:
        json.dump(vocab_dict, vocab_file)
    
    return file_dict