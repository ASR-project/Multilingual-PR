import torch
from torch.nn.utils.rnn import pad_sequence

def coll_fn(batch):
    batch_dict={}

    list_tensors = [torch.from_numpy(b['audio']['array']) for b in batch] # torch.from_numpy costly
    batch_audio = pad_sequence(list_tensors, padding_value=-10, batch_first=True)
    batch_dict['array'] = batch_audio
    #batch_audio = F.pad(([torch.from_numpy(b['audio']['array']) for b in batch]), pad=-100)
    # TODO padding for labels
    # TODO add .mp3 to disply on wand
    return batch_dict