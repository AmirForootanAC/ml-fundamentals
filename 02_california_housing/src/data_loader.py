# =========================
# description: load california housing dataset
# =========================

from sklearn.datasets import fetch_california_housing
import pandas as pd

def load_data():
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    return df