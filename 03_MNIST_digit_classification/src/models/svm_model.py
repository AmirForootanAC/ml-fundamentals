from sklearn.svm import SVC


class SVMModel:

    def train(
        self,
        X_train,
        y_train
    ):

        model = SVC(
            kernel="rbf",
            random_state=42
        )

        model.fit(
            X_train,
            y_train
        )

        return model