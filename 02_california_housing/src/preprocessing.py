import numpy as np
import pandas as pd

# =========================
# description: feature engineering for california housing
# =========================

def create_features(df):
    df = df.copy()

    # Rooms per household
    df["RoomsPerHousehold"] = (
        df["AveRooms"] / df["AveOccup"]
    )

    # Bedrooms ratio
    df["BedroomsRatio"] = (
        df["AveBedrms"] / df["AveRooms"]
    )

    # Population per household
    df["PopulationPerHousehold"] = (
        df["Population"] / df["AveOccup"]
    )

    return df