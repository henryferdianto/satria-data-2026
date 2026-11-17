from sklearn.ensemble import RandomForestClassifier

from src.features import build_features
from src.sample_data import FEATURES, make


def test_sample_data_shape():
    df = make(200)
    assert len(df) == 200
    assert set(FEATURES) <= set(df.columns)
    assert set(df.target.unique()) <= {0, 1}


def test_features_and_model_fit():
    df = make(300)
    X, y = build_features(df, "target")
    model = RandomForestClassifier(n_estimators=50, random_state=0).fit(X, y)
    assert model.predict(X).shape == (300,)
