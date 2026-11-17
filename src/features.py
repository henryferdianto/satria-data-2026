"""Feature engineering for the SATRIA DATA modelling pipeline."""
import pandas as pd
from sklearn.preprocessing import StandardScaler


def build_features(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target]
    num = X.select_dtypes("number").columns
    X[num] = StandardScaler().fit_transform(X[num])
    return pd.get_dummies(X, drop_first=True), y
