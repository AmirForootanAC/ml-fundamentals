import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# =========================
# description: basic plotting utilities for EDA
# =========================

def plot_distribution(df, col):
    plt.figure(figsize=(8,5))
    sns.histplot(df[col], bins=50)
    plt.title(f"Distribution of {col}")
    plt.show()


def plot_scatter(df, x, y):
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df[x], y=df[y])
    plt.title(f"{x} vs {y}")
    plt.show()

# =========================
# description: correlation heatmap
# =========================

def plot_correlation(df):
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
    plt.title("Feature Correlation Heatmap")
    plt.show()

# =========================
# description: spatial visualization
# =========================

def plot_geography(df):
    plt.figure(figsize=(8,6))
    plt.scatter(
        df["Longitude"],
        df["Latitude"],
        c=df["MedHouseVal"],
        cmap="viridis",
        s=10
    )
    plt.colorbar(label="House Value (×100k)")
    plt.title("California Housing Price Map")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

# =========================
# description: feature importance plot
# =========================

def plot_feature_importance(model, X, top_n=15):

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    plt.figure(figsize=(10,6))

    sns.barplot(
        data=importance_df.head(top_n),
        x="Importance",
        y="Feature"
    )

    plt.title(
        f"Top {top_n} Feature Importances"
    )

    plt.show()

    return importance_df

# =========================
# description: actual vs predicted plot
# =========================

import matplotlib.pyplot as plt


def plot_actual_vs_predicted(
    y_true,
    y_pred
):

    plt.figure(
        figsize=(8,8)
    )

    plt.scatter(
        y_true,
        y_pred,
        alpha=0.5
    )

    plt.xlabel(
        "Actual Values"
    )

    plt.ylabel(
        "Predicted Values"
    )

    plt.title(
        "Actual vs Predicted"
    )

    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        "r--"
    )

    plt.show()

# =========================
# description: residual plot
# =========================

import matplotlib.pyplot as plt


def plot_residuals(
    y_true,
    y_pred
):

    residuals = y_true - y_pred

    plt.figure(
        figsize=(8,6)
    )

    plt.scatter(
        y_pred,
        residuals,
        alpha=0.5
    )

    plt.axhline(
        y=0,
        color="red",
        linestyle="--"
    )

    plt.xlabel(
        "Predicted Values"
    )

    plt.ylabel(
        "Residuals"
    )

    plt.title(
        "Residual Plot"
    )

    plt.show()