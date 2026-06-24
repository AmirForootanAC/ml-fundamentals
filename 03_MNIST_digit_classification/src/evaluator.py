from sklearn.metrics import accuracy_score


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