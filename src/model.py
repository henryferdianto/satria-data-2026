"""Train and cross-validate candidate models."""
import argparse
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def candidates():
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "rf": RandomForestClassifier(n_estimators=400, random_state=42),
        "xgb": XGBClassifier(n_estimators=400, learning_rate=0.05, random_state=42),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--target", default="target")
    args = ap.parse_args()
    df = pd.read_csv(args.train)
    X, y = df.drop(columns=[args.target]), df[args.target]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best, best_score = None, -1
    for name, model in candidates().items():
        score = cross_val_score(model, X, y, cv=cv, scoring="f1_macro").mean()
        print(f"{name}: macro-F1 = {score:.4f}")
        if score > best_score:
            best, best_score = model, score

    best.fit(X, y)
    joblib.dump(best, "models/best_model.joblib")
    print(f"best model saved (macro-F1 {best_score:.4f})")


if __name__ == "__main__":
    main()
