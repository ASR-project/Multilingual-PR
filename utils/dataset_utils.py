import torch
from torch.nn.utils.rnn import pad_sequence
import os
import os.path as osp
import json

import pandas as pd
from utils.logger import init_logger

def coll_fn(batch):
    
    batch_dict={}
    batch_dict['array'] = pad_sequence([torch.Tensor(b['audio']) for b in batch], padding_value=0, batch_first=True)
    batch_dict['path'] = [b['path'] for b in batch]
    batch_dict['sentence'] = [b['sentence'] for b in batch]
    # batch_dict['labels'] = [b['labels'] for b in batch]
    
    return batch_dict

# def coll_fn(batch):
#     batch_dict={}
#     batch_dict['array'] = pad_sequence([torch.Tensor(b['audio']['array']) for b in batch], padding_value=0, batch_first=True)
#     batch_dict['path'] = [b['path'] for b in batch]
#     batch_dict['sentence'] = [b['sentence'] for b in batch]
    
#     return batch_dict

def create_vocabulary(ISO6393, path_csv, eos_token, bos_token, unk_token, pad_token, word_delimiter_token):

    logger = init_logger("create_vocabulary", "INFO")

    df = pd.read_csv(osp.join(path_csv, "phoible.csv"))

    df_phoneme_target_lang = df[df['ISO6393']==ISO6393]['Phoneme']
    df_phoneme_target_lang.drop_duplicates(keep="first", inplace=True)
    df_phoneme_target_lang.reset_index(drop=True, inplace=True)

    phoneme_vocab = dict(df_phoneme_target_lang)
    phoneme_vocab = {v:k for k,v in phoneme_vocab.items()}

    phoneme_vocab[eos_token] = len(phoneme_vocab)
    phoneme_vocab[bos_token] = len(phoneme_vocab)
    phoneme_vocab[unk_token] = len(phoneme_vocab)
    phoneme_vocab[pad_token] = len(phoneme_vocab)
    phoneme_vocab[word_delimiter_token] = len(phoneme_vocab)

    logger.info(f'Length vocabulary : {len(phoneme_vocab)}')

    vocab_path = osp.join(os.getcwd(), "assets", "vocab_phoneme")
    file_dict = os.path.join(vocab_path, f"vocab-phoneme-{ISO6393}.json")

    if not os.path.exists(vocab_path):
        os.makedirs(vocab_path)

    with open(file_dict, "w") as vocab_file:
        json.dump(phoneme_vocab, vocab_file)
    
    return file_dict, len(phoneme_vocab)


def create_vocabulary2(language, path, eos_token, bos_token, unk_token, pad_token, word_delimiter_token):

    logger = init_logger("create_vocabulary", "INFO")

    json_file = osp.join(path, language, "phonesMatches_reduced.json")

    with open(json_file) as file:
        phoneme_vocab = json.load(file)

    phoneme_vocab[eos_token] = len(phoneme_vocab)
    phoneme_vocab[bos_token] = len(phoneme_vocab)
    phoneme_vocab[unk_token] = len(phoneme_vocab)
    phoneme_vocab[pad_token] = len(phoneme_vocab)
    phoneme_vocab[word_delimiter_token] = len(phoneme_vocab)

    logger.info(f'Length vocabulary : {len(phoneme_vocab)}')

    vocab_path = osp.join(os.getcwd(), "assets", "vocab_phoneme")
    file_dict = os.path.join(vocab_path, f"vocab-phoneme-{language}.json")

    if not os.path.exists(vocab_path):
        os.makedirs(vocab_path)

    with open(file_dict, "w") as vocab_file:
        json.dump(phoneme_vocab, vocab_file)
    
    return file_dict, len(phoneme_vocab)