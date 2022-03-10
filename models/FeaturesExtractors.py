import torch.nn as nn
from transformers import (HubertForCTC, Wav2Vec2FeatureExtractor,
                          Wav2Vec2ForCTC,
                          Wav2Vec2Processor, WavLMForCTC)


class BaseFeaturesExtractor(nn.Module):
    """
        BaseFeaturesExtractor class that will extract features according to the type of model
        https://huggingface.co/blog/fine-tune-wav2vec2-english
    """

    def __init__(self, params):
        super().__init__()
        self.params = params

    def forward(self, x):
        # input_values = self.processor(x, sampling_rate=self.params.sampling_rate, return_tensors="pt").input_values.squeeze(0)
        # features = self.processor.feature_extractor(input_values, return_tensors="pt").input_values.squeeze(0)
        # features2 = self.feature_extractor2(x, return_tensors="pt").input_values
        # print(features2.shape)
        # FIXME extractor features correctly
        logits = self.model(x).logits
        # features = self.feature_extractor(x) 
        # print(features.shape)
        # features_projected = self.feature_projection(features)
        # features_encoded = self.encoder(features_projected)
        return logits

class Wav2Vec2(BaseFeaturesExtractor):
    """
    https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ForCTC
    """

    def __init__(self, params):
        super().__init__(params)
        self.model = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft", feature_size = 1, sampling_rate= 16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
        # self.model = Wav2Vec2FeatureExtractor.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus", feature_size = 1, sampling_rate= 16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

class WavLM(BaseFeaturesExtractor):
    """
    https://huggingface.co/docs/transformers/model_doc/wavlm#transformers.WavLMForCTC
    """

    def __init__(self, params):
        super().__init__(params)
        self.processor = Wav2Vec2Processor.from_pretrained(
            "patrickvonplaten/wavlm-libri-clean-100h-base-plus")
        self.model = WavLMForCTC.from_pretrained(
            "patrickvonplaten/wavlm-libri-clean-100h-base-plus").wavlm.feature_extractor


class Hubert(BaseFeaturesExtractor):
    """
    https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/hubert#transformers.HubertForCTC
    """

    def __init__(self, params):
        super().__init__(params)
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/hubert-large-ls960-ft")
        self.model = HubertForCTC.from_pretrained(
            "facebook/hubert-large-ls960-ft").hubert.feature_extractor
