import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


class MNISTVisualizer:

    def show_samples(
        self,
        X,
        y,
        num_samples=10
    ):

        plt.figure(figsize=(10, 5))

        for i in range(num_samples):

            plt.subplot(2, 5, i + 1)

            image = X[i].reshape(28, 28)

            plt.imshow(
                image,
                cmap="gray"
            )

            plt.title(f"Label: {y[i]}")

            plt.axis("off")

        plt.tight_layout()

        plt.show()

    def show_confusion_matrix(
        self,
        y_true,
        y_pred
    ):

        ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred
        )

        plt.title(
            "MNIST Confusion Matrix"
        )

        plt.show()