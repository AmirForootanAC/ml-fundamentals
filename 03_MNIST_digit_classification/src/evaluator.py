from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


class ModelEvaluator:

    def accuracy(
        self,
        y_true,
        y_pred
    ):
        return accuracy_score(
            y_true,
            y_pred
        )

    def classification_report(
        self,
        y_true,
        y_pred
    ):

        return classification_report(
            y_true,
            y_pred
        )

    def confusion_matrix(
        self,
        y_true,
        y_pred
    ):

        return confusion_matrix(
            y_true,
            y_pred
        )