import numpy as np


class MNISTPreprocessor:

    def normalize(self, X):

        X = X.astype(np.float32)

        return X / 255.0