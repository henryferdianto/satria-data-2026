"""Generate a synthetic development dataset for the pipeline."""
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "region_income_index",
    "education_index",
    "unemployment_rate",
    "internet_penetration",
    "urban_share",
]


def make(n: int = 800, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {f: rng.normal(0, 1, n) for f in FEATURES}
    data["region_income_index"] = rng.uniform(20, 90, n)
    data["education_index"] = rng.uniform(0.3, 0.95, n)
    data["unemployment_rate"] = rng.uniform(2, 15, n)
    data["internet_penetration"] = rng.uniform(0.2, 0.99, n)
    data["urban_share"] = rng.uniform(0.1, 0.95, n)

    score = (
        0.4 * (data["region_income_index"] / 90)
        + 0.3 * data["education_index"]
        + 0.2 * data["internet_penetration"]
        + 0.1 * data["urban_share"]
        - 0.2 * (data["unemployment_rate"] / 15)
        + rng.normal(0, 0.05, n)
    )
    data["target"] = (score > np.median(score)).astype(int)
    return pd.DataFrame(data)


def main() -> Path:
    out = Path("data/raw/train.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df = make()
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} rows -> {out}")
    return out


if __name__ == "__main__":
    main()
