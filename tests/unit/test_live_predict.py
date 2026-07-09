
from storm_eval.forecasting.dataset import generate_dataset
from storm_eval.forecasting.features import build_training_table
from storm_eval.forecasting.model import train
from storm_eval.live.predict import estimate_landfall


def _model():
    X, y, _ = build_training_table(generate_dataset(n_storms=300, seed=4))
    return train(X, y)


def test_estimate_returns_category_for_valid_track():
    track = [
        {"wind": 90, "pres": 965, "lat": 28.0, "lon": -79.0},
        {"wind": 95, "pres": 960, "lat": 30.0, "lon": -78.0},
        {"wind": 92, "pres": 962, "lat": 31.5, "lon": -77.5},
        {"wind": 85, "pres": 968, "lat": 32.5, "lon": -77.2},
        {"wind": 80, "pres": 972, "lat": 33.2, "lon": -77.0},
    ]
    est = estimate_landfall(track, _model())
    assert est.category in {0, 1, 2, 3, 4, 5}
    assert "not an official" in est.disclaimer.lower()


def test_estimate_none_for_short_track():
    est = estimate_landfall([{"wind": 80, "pres": 980, "lat": 30, "lon": -78}], _model())
    assert est.category is None
