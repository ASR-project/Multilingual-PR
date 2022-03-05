import importlib
import numpy as np
import os

from config.hparams import Parameters


def test_features():
    # load hparams
    params = Parameters.parse()

    # load the right class
    network_name = params.feat_param.network_name
    mod = importlib.import_module(f"models.FeaturesExtractors")
    feat_ext_class = getattr(mod, network_name)(params)

    # run a dummy feature extraction
    dummy_input = np.array([])
    path_features = feat_ext_class.extract_features(dummy_input)
    assert os.path.exists(path_features)

    loaded_features = np.load(open(path_features, 'rb'))
    assert len(loaded_features) == 0
