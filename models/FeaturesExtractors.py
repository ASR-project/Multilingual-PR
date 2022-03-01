import torch.nn as nn
from transformers import (HubertForCTC, Wav2Vec2ForCTC, Wav2Vec2Processor, WavLMForCTC)

class BaseFeaturesExtractor(nn.Module):
    """
        BaseFeaturesExtractor class that will extract features according to the type of model
        https://huggingface.co/blog/fine-tune-wav2vec2-english
    """
    def __init__(self, params):
        super().__init__()
        self.params = params

    def forward(self, x):
        x = self.processor(x, return_tensors="pt").input_values.squeeze(0)
        output = self.model(x)
        return output

class Wav2Vec2(BaseFeaturesExtractor):
    """
    https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ForCTC
    """
    def __init__(self, params):
        super().__init__(params)
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").wav2vec2.feature_extractor

class WavLM(BaseFeaturesExtractor):
    """
    https://huggingface.co/docs/transformers/model_doc/wavlm#transformers.WavLMForCTC
    """
    def __init__(self, params):
        super().__init__(params)
        self.processor = Wav2Vec2Processor.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")
        self.model = WavLMForCTC.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus").wavlm.feature_extractor

class Hubert(BaseFeaturesExtractor):
    """
    https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/hubert#transformers.HubertForCTC
    """
    def __init__(self, params):
        super().__init__(params)
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
        self.model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft").hubert.feature_extractor
