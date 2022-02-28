import torch
from torch.nn.utils.rnn import pad_sequence

def coll_fn(batch):
    list_tensors = [torch.from_numpy(b['audio']['array']) for b in batch]
    batch_audio = pad_sequence(list_tensors, padding_value=-10, batch_first=True)
    #batch_audio = F.pad(([torch.from_numpy(b['audio']['array']) for b in batch]), pad=-100)
    # TODO padding for labels
    return batch_audio
    