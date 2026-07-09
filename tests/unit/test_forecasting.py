import numpy as np

from storm_eval.forecasting.dataset import generate_dataset
from storm_eval.forecasting.features import FEATURE_NAMES, build_training_table
from storm_eval.forecasting.evaluate import evaluate, year_split


def test_feature_table_shape():
    storms = generate_dataset(n_storms=120, seed=5)
    X, y, years = build_training_table(storms)
    assert X.shape[1] == len(FEATURE_NAMES)
    assert len(X) == len(y) == len(years) > 20
    assert set(np.unique(y)) <= {0, 1, 2, 3, 4, 5}


def test_year_split_no_leakage():
    years = np.array([2000, 2000, 2001, 2002, 2003, 2003])
    tr, te = year_split(years, test_frac=0.5, seed=1)
    train_years = set(years[tr]); test_years = set(years[te])
    assert train_years.isdisjoint(test_years)


def test_model_learns_signal():
    # On structured synthetic data the model must beat majority-class baseline.
    storms = generate_dataset(n_storms=500, seed=11)
    report, _ = evaluate(storms)
    _, y, _ = build_training_table(storms)
    majority = np.bincount(y).max() / len(y)
    assert report.accuracy > majority
    assert report.within_one >= report.accuracy
