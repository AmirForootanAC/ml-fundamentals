from src.data_loader import MNISTDataLoader
from src.visualizer import MNISTVisualizer


def main():

    loader = MNISTDataLoader()
    visualizer = MNISTVisualizer()

    X, y = loader.load()

    visualizer.show_samples(X, y)


if __name__ == "__main__":
    main()