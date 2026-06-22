# =========================
# description: baseline regression models
# =========================

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def get_linear_model():

    return LinearRegression()


def get_random_forest():

    return RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )