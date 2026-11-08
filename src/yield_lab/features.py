"""Features computed only from the current and earlier calendar months."""
import numpy as np
import pandas as pd
from .data import LABELS

ML_FEATURE_NAMES = (
    ["level_" + x for x in LABELS]
    + ["lag1_" + x for x in LABELS]
    + ["lag3_" + x for x in LABELS]
    + ["change1_" + x for x in LABELS]
    + ["change3_" + x for x in LABELS]
    + ["slope_30y_28d", "slope_10y_3y"]
)


def ml_features(causal_panel):
    level = causal_panel
    lag1, lag3 = level.shift(1), level.shift(3)
    frame = pd.concat([
        level, lag1, lag3, level - lag1, level - lag3,
        (level.iloc[:, 8] - level.iloc[:, 0]).to_frame(),
        (level.iloc[:, 6] - level.iloc[:, 4]).to_frame(),
    ], axis=1)
    frame.columns = ML_FEATURE_NAMES
    return frame


def scenario_features(latest_features, rates_pct):
    """Replace contemporaneous rates, retaining the stored historical lags."""
    features = np.asarray(latest_features, dtype=float).copy()
    current = np.asarray(rates_pct, dtype=float)
    if current.shape != (9,) or not np.isfinite(current).all():
        raise ValueError("rates_pct must contain exactly nine finite numbers")
    features[:9] = current
    features[27:36] = current - features[9:18]
    features[36:45] = current - features[18:27]
    features[45:] = [current[8] - current[0], current[6] - current[4]]
    return features
