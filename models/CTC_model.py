import torch.nn as nn
import torch.nn.functional as F
import torch

class CTC_model(nn.Module):
    """
    https://codingvision.net/pytorch-crnn-seq2seq-digits-recognition-ctc
    """
    def __init__(self, params) -> None:
        super().__init__()
        num_features = 512
        num_classes = 100 # to check
        
        num_layers = 2
        hidden_size = 256
        
        self.gru_input_size = 3 * 32
        gru_hidden_size = 128
        gru_num_layers = 2
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3))
        self.in1 = nn.InstanceNorm2d(32)

        self.gru = nn.GRU(
            self.gru_input_size, 
            gru_hidden_size, 
            gru_num_layers, 
            batch_first = True, 
            bidirectional = True
        )

        self.fc = nn.Linear(gru_hidden_size * 2, num_classes)
        
    def forward(self, x):
        batch_size = x.shape[0]
            
        out = self.conv1(x.unsqueeze(1)) 
        out = F.leaky_relu(out)
        out = self.in1(out)

        out = out.permute(0, 3, 2, 1) 
        out = out.reshape(batch_size, -1, self.gru_input_size)

        out, gru_h = self.gru(out, None)
        self.gru_h = gru_h.detach()
        out = torch.stack([F.log_softmax(self.fc(out[i])) for i in range(out.shape[0])])

        return out