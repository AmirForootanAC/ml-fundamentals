import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

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

    def show_misclassified_samples(
        self,
        X_test,
        y_test,
        predictions,
        num_samples=9
    ):

        misclassified = (
            predictions != y_test
        )

        X_errors = X_test[
            misclassified
        ]

        y_true = y_test[
            misclassified
        ]

        y_pred = predictions[
            misclassified
        ]

        plt.figure(
            figsize=(8, 8)
        )

        for i in range(
            min(
                num_samples,
                len(X_errors)
            )
        ):

            plt.subplot(
                3,
                3,
                i + 1
            )

            plt.imshow(
                X_errors[i].reshape(
                    28,
                    28
                ),
                cmap="gray"
            )

            plt.title(
                f"T:{y_true[i]} "
                f"P:{y_pred[i]}"
            )

            plt.axis("off")

        plt.tight_layout()

        plt.show()