from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# =========================
# description: baseline regression models
# =========================

def get_linear_model():

    return LinearRegression()


def get_random_forest():

    return RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

# =========================
# description: xgboost regressor
# =========================

def get_xgboost():

    return XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )