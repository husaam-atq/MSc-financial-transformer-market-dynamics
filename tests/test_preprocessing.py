from __future__ import annotations

import numpy as np
import pandas as pd

from market_dynamics.preprocessing.scaling import TrainOnlyPreprocessor


def test_preprocessing_fits_statistics_on_train_only() -> None:
    X_train = pd.DataFrame({"x": [1.0, 2.0, 3.0], "z": [10.0, 10.0, 10.0]})
    X_test = pd.DataFrame({"x": [1000.0], "z": [999.0]})

    preprocessor = TrainOnlyPreprocessor(scale=True).fit(X_train)
    transformed_test = preprocessor.transform(X_test)

    assert np.isclose(preprocessor.imputer_statistics_[0], 2.0)
    assert np.isclose(preprocessor.scaler_mean_[0], 2.0)
    assert transformed_test.shape == (1, 2)


def test_preprocessing_output_shapes() -> None:
    X_train = pd.DataFrame({"x": [1.0, np.nan, 3.0], "z": [4.0, 5.0, 6.0]})
    preprocessor = TrainOnlyPreprocessor(scale=False)
    transformed = preprocessor.fit_transform(X_train)

    assert transformed.shape == X_train.shape
    assert not np.isnan(transformed).any()


def test_preprocessing_retains_feature_that_is_all_missing_in_training() -> None:
    train = pd.DataFrame({"observed": [1.0, 2.0], "late_macro": [np.nan, np.nan]})
    evaluation = pd.DataFrame({"observed": [3.0], "late_macro": [5.0]})
    preprocessor = TrainOnlyPreprocessor(scale=True).fit(train)

    transformed = preprocessor.transform(evaluation)

    assert transformed.shape == (1, 2)
    assert np.isfinite(transformed).all()
