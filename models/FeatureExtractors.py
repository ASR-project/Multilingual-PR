import os

import numpy as np
import wandb
from transformers import Wav2Vec2FeatureExtractor

from models.BaseFeatureExtractor import BaseFeatureExtractor


class Wav2vec(BaseFeatureExtractor):
    def __init__(self, params):
        super().__init__(params)

    def extract_features(self, dataset):
        """
            https://huggingface.co/blog/fine-tune-wav2vec2-english
        """
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=self.params.feat_param.feature_size,
                                                     sampling_rate=self.params.feat_param.sampling_rate, 
                                                     padding_value=self.params.feat_param.padding_value, 
                                                     do_normalize=self.params.feat_param.do_normalize, 
                                                     return_attention_mask=self.params.feat_param.return_attention_mask)

        # TODO: for loop to extract features
        features = np.array([])

        # Get the path to the features and save them
        # name of the features file
        name_features = '-'.join([self.__class__.__name__.lower(),
                                 self.params.data_param.subset])
        path_features = os.path.join(
            self.params.feat_param.path_features, name_features+'.npy')
        np.save(
            open(path_features, 'wb'),
            features
        )
        return path_features

    def push_artifact(self, path_features):
        artifact_name = self.__class__.__name__.lower()
        artifact = wandb.Artifact(
            name=artifact_name,
            type='dataset',
            # metadata=dict(self.params.feat_param) #FIXME
        )
        artifact.add_file(path_features)
        wandb.log_artifact(artifact, aliases=["latest"])