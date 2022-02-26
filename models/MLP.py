from math import log

import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, params) -> None:
        super().__init__()
        in_size = params.input_size

        self.norm = getattr(nn, params.normalization)
        self.activation = getattr(nn, params.activation)

        self.layers = nn.ModuleList()

        self.layers.append(nn.Sequential(nn.Linear(in_size, 2*in_size),
                                         self.norm(2*in_size),
                                         self.activation(),
                                         nn.Dropout(params.dropout)
                                         )
                           )
        self.layers.append(nn.Sequential(nn.Linear(2*in_size, in_size),
                                         self.norm(in_size),
                                         self.activation(),
                                         nn.Dropout(params.dropout)
                                         )
                           )

        for i in range(int(log(in_size, 2))):
            self.layers.append(nn.Sequential(nn.Linear(in_size, in_size//2),
                                             self.norm(in_size//2),
                                             self.activation(),
                                             nn.Dropout(params.dropout)
                                             )
                               )

            in_size //= 2

            if in_size <= 400:
                break

        self.layers.append(nn.Linear(in_size, 1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.squeeze()
