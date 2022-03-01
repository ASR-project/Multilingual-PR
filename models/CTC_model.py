import torch
import torch.nn as nn
import torch.nn.functional as F


class CTC_model(nn.Module):
    """
    https://codingvision.net/pytorch-crnn-seq2seq-digits-recognition-ctc
    """

    def __init__(self, params) -> None:
        super().__init__()
        
        self.num_classes = params.len_vocab

        self.gru_input_size = 512
        self.gru_hidden_size = 128
        self.gru_num_layers = 2
        self.gru_h = None
        self.gru_cell = None

        self.gru = nn.GRU(self.gru_input_size, self.gru_hidden_size,
                          self.gru_num_layers, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(self.gru_hidden_size * 2, self.num_classes)

    def forward(self, x):
        batch_size = x.shape[0]

        out = x.reshape(batch_size, -1, self.gru_input_size)

        out, gru_h = self.gru(out, None)
        self.gru_h = gru_h.detach()
        out = torch.stack([F.log_softmax(self.fc(out[i]))
                          for i in range(out.shape[0])])

        return out
