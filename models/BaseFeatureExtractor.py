class BaseFeatureExtractor(object):
    """
        BaseFeatureExtractor class that will extract features according to the type of model
    """
    def __init__(self, params):
        self.params = params

    def extract_features(self, input):
        raise NotImplementedError(f'Should be implemented in derived class!')
    
    def push_artifact(self, features):
        raise NotImplementedError(f'Should be implemented in derived class!')
    

