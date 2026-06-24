import matplotlib.pyplot as plt


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