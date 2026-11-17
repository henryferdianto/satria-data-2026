"""End-to-end demo on synthetic data."""
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.model import train
from src.sample_data import make


def main():
    df = make()
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df.target, random_state=42)
    model, score = train(train_df)
    pred = model.predict(test_df.drop(columns=["target"]))
    print(classification_report(test_df.target, pred))
    print(f"hold-out macro-F1: {score:.4f} (cv)")


if __name__ == "__main__":
    main()
