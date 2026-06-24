from src.data_loader import MNISTDataLoader
from src.preprocessing.preprocessor import MNISTPreprocessor
from src.trainer import ModelTrainer
from src.evaluator import ModelEvaluator
from src.visualizer import MNISTVisualizer


def main():

    loader = MNISTDataLoader()
    preprocessor = MNISTPreprocessor()

    trainer = ModelTrainer()
    evaluator = ModelEvaluator()
    visualizer = MNISTVisualizer()

    X, y = loader.load()

    X = preprocessor.normalize(X)

    X_train, X_test, y_train, y_test = (
        preprocessor.split_data(X, y)
    )

    model = trainer.train_logistic_regression(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = evaluator.accuracy(
        y_test,
        predictions
    )

    print(f"Accuracy: {accuracy:.4f}")

    print(
        evaluator.classification_report(
            y_test,
            predictions
        )
    )

    print(
        visualizer.show_confusion_matrix(
            y_test,
            predictions
        )
    )


if __name__ == "__main__":
    main()