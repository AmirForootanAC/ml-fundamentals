from sklearn.datasets import fetch_openml


class MNISTDataLoader:

    def load(self):

        mnist = fetch_openml(
            "mnist_784",
            version=1,
            as_frame=False
        )

        X = mnist.data
        y = mnist.target.astype(int)

        return X, y