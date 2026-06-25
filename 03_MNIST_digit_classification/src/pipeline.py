from src.data_loader import MNISTDataLoader
from src.preprocessing.preprocessor import MNISTPreprocessor
from src.models.logistic_regression_model import (LogisticRegressionModel)
from src.models.random_forest_model import (RandomForestModel)
from src.models.svm_model import SVMModel
from src.evaluator import ModelEvaluator
from src.visualizer import MNISTVisualizer


class MNISTPipeline:

    def __init__(self):

        self.loader = MNISTDataLoader()

        self.preprocessor = (
            MNISTPreprocessor()
        )

        self.model = (
            SVMModel()
        )

        self.evaluator = (
            ModelEvaluator()
        )

        self.visualizer = (
            MNISTVisualizer()
        )

    def run(self):

        # =====================
        # Load Data
        # =====================

        X, y = self.loader.load()

        # =====================
        # Preprocessing
        # =====================

        X = self.preprocessor.normalize(X)

        (
            X_train,
            X_test,
            y_train,
            y_test
        ) = self.preprocessor.split_data(
            X,
            y
        )

        # =====================
        # Training
        # =====================

        trained_model = (
            self.model.train(
                X_train,
                y_train
            )
        )

        # =====================
        # Prediction
        # =====================

        predictions = (
            trained_model.predict(
                X_test
            )
        )

        # =====================
        # Evaluation
        # =====================

        accuracy = (
            self.evaluator.accuracy(
                y_test,
                predictions
            )
        )

        print(
            f"\nAccuracy: {accuracy:.4f}\n"
        )

        print(
            self.evaluator
            .classification_report(
                y_test,
                predictions
            )
        )

        # =====================
        # Visualization
        # =====================

        self.visualizer.show_confusion_matrix(
            y_test,
            predictions
        )