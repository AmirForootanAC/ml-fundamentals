import pandas as pd


class MNISTAnalyzer:

    def get_dataset_info(self, X, y):

        print("=" * 50)
        print("Dataset Information")
        print("=" * 50)

        print(f"Number of samples : {X.shape[0]}")
        print(f"Number of features: {X.shape[1]}")
        print(f"Number of classes : {len(set(y))}")

        print("\nPixel Range:")
        print(f"Min Value: {X.min()}")
        print(f"Max Value: {X.max()}")

    def get_class_distribution(self, y):

        distribution = (
            pd.Series(y)
            .value_counts()
            .sort_index()
        )

        return distribution