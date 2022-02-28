from importlib_metadata import SelectableGroups
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

class wav2vec2(nn.Module):
    def __init__(self, params):
        super().__init__()
        
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").wav2vec2.feature_extractor

    def forward(self, x):
        x = self.processor(x, return_tensors="pt").input_values.squeeze(0)
        output = self.model(x)
        return output