import os
import joblib


class ModelSaver:

    def save(
        self,
        model,
        path
    ):

        os.makedirs(
            os.path.dirname(path),
            exist_ok=True
        )

        joblib.dump(
            model,
            path
        )

        print(
            f"Model saved to: {path}"
        )

    def load(
        self,
        path
    ):

        return joblib.load(
            path
        )