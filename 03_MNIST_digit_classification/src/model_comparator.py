class ModelComparator:

    def __init__(self):

        self.results = []

    def add_result(
        self,
        model_name,
        accuracy
    ):

        self.results.append(
            {
                "model": model_name,
                "accuracy": accuracy
            }
        )

    def show_results(self):

        if not self.results:

            print("No results available.")
            return

        sorted_results = sorted(
            self.results,
            key=lambda x: x["accuracy"],
            reverse=True
        )

        print("\n" + "=" * 55)
        print("MODEL COMPARISON")
        print("=" * 55)

        print(
            f"{'Rank':<6}"
            f"{'Model':<25}"
            f"{'Accuracy':<12}"
        )

        print("-" * 55)

        for rank, result in enumerate(
            sorted_results,
            start=1
        ):

            print(
                f"{rank:<6}"
                f"{result['model']:<25}"
                f"{result['accuracy']:.4f}"
            )

        best_model = sorted_results[0]
        baseline = sorted_results[-1]

        improvement = (
            best_model["accuracy"]
            - baseline["accuracy"]
        )

        print("\n" + "=" * 55)

        print(
            f"Best Model : "
            f"{best_model['model']}"
        )

        print(
            f"Accuracy   : "
            f"{best_model['accuracy']:.4f}"
        )

        print(
            f"Improvement Over Baseline : "
            f"{improvement:.4f}"
        )

        print("=" * 55)