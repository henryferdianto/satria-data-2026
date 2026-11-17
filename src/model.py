"""Train and cross-validate candidate models (scikit-learn only)."""
import argparse

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def candidates() -> dict:
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "rf": RandomForestClassifier(n_estimators=400, random_state=42),
        "histgb": HistGradientBoostingClassifier(random_state=42),
    }


def train(df: pd.DataFrame, target: str = "target"):
    X, y = df.drop(columns=[target]), df[target]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best, best_score = None, -1.0
    for name, model in candidates().items():
        score = cross_val_score(model, X, y, cv=cv, scoring="f1_macro").mean()
        print(f"{name}: macro-F1 = {score:.4f}")
        if score > best_score:
            best, best_score = model, score
    best.fit(X, y)
    return best, best_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--target", default="target")
    args = ap.parse_args()
    model, score = train(pd.read_csv(args.train), args.target)
    joblib.dump(model, "models/best_model.joblib")
    print(f"best model saved (macro-F1 {score:.4f})")


if __name__ == "__main__":
    main()
