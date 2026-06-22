import seaborn as sns
import matplotlib.pyplot as plt

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