"""Load and clean raw SATRIA DATA competition datasets."""
import argparse
import pandas as pd


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df.drop_duplicates()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes("number").columns
    cat_cols = df.select_dtypes("object").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna("unknown")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    df = clean(load(args.input))
    df.to_csv(args.output, index=False)
    print(f"saved {len(df)} rows -> {args.output}")


if __name__ == "__main__":
    main()
