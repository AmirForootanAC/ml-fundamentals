from src.data_loader import MNISTDataLoader
from src.preprocessing.preprocessor import MNISTPreprocessor


def main():

    loader = MNISTDataLoader()
    preprocessor = MNISTPreprocessor()

    X, y = loader.load()

    print("Before Normalization")
    print(X.min(), X.max())

    X = preprocessor.normalize(X)

    print("After Normalization")
    print(X.min(), X.max())


if __name__ == "__main__":
    main()