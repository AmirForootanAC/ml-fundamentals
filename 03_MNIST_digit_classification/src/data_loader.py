import pandas as pd


class MNISTDataLoader:

    def load(self):

        df = pd.read_csv(
            "data/mnist_train.csv"
        )

        X = df.drop(
            columns=["label"]
        ).values

        y = df["label"].values

        return X, y