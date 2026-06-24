from sklearn.linear_model import LogisticRegression


class LogisticRegressionModel:

    def train(
        self,
        X_train,
        y_train
    ):

        model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )

        model.fit(
            X_train,
            y_train
        )

        return model