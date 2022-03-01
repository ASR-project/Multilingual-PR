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
        features = self.feature_extractor(x)
        print(features.shape)
        features2 = self.feature_extractor2(x, return_tensors="pt").input_values
        print(features2.shape)
        return features

class Wav2Vec2(BaseFeaturesExtractor):
    """
    https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ForCTC
    """

    def __init__(self, params):
        super().__init__(params)

        self.feature_extractor2 = Wav2Vec2FeatureExtractor(feature_size=params.feature_size,
                                                    sampling_rate=params.sampling_rate, 
                                                    padding_value=params.padding_value, 
                                                    do_normalize=params.do_normalize,
                                                    return_attention_mask=params.return_attention_mask)

        # self.processor = Wav2Vec2Processor(feature_extractor=feature_extractor)
        self.feature_extractor = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").wav2vec2.feature_extractor
        # self.model.lm_head = nn.Linear(in_features=768, out_features=tokenizer.vocab_size)

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
