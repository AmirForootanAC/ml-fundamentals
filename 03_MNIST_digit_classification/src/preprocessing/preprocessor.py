from sklearn.model_selection import train_test_split
import numpy as np


class MNISTPreprocessor:

    def normalize(self, X):

        X = X.astype(np.float32)

        return X / 255.0

    def split_data(
        self,
        X,
        y,
        test_size=0.2,
        random_state=42
    ):

        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )