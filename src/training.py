"""Chronological supervision, train-only scaling and nested epoch selection."""
import numpy as np
from yield_lab.data import causal_features
from yield_lab.features import ml_features
from network import DenseMLP, SETTINGS

HORIZONS = (3, 12)


def supervision(panel, origin):
    history = panel.iloc[:origin + 1]
    features = ml_features(causal_features(history)).to_numpy(dtype=float)
    raw = history.to_numpy(dtype=float)
    targets = np.full((len(history), 18), np.nan)
    for k, horizon in enumerate(HORIZONS):
        n = len(history) - horizon
        if n > 0:
            targets[:n, k * 9:(k + 1) * 9] = raw[horizon:] - features[:n, :9]
    eligible = np.isfinite(features[:, :9]).all(axis=1) & np.isfinite(targets).any(axis=1)
    return features[eligible], targets[eligible], np.flatnonzero(eligible), features[-1]


def fit_scalers(x, y):
    median = np.nanmedian(x, axis=0)
    if not np.isfinite(median).all():
        raise ValueError("Not enough history to estimate every feature")
    filled = np.where(np.isfinite(x), x, median)
    xmean, xscale = filled.mean(axis=0), filled.std(axis=0)
    xscale[xscale < 1e-8] = 1.
    ymean, yscale = np.nanmean(y, axis=0), np.nanstd(y, axis=0)
    if not np.isfinite(ymean).all():
        raise ValueError("Some neural outputs have no observed training targets")
    yscale[yscale < .1] = .1
    return {"x_median": median, "x_mean": xmean, "x_scale": xscale,
            "y_mean": ymean, "y_scale": yscale}


def transform(x, y, scalers):
    filled = np.where(np.isfinite(x), x, scalers["x_median"])
    zx = (filled - scalers["x_mean"]) / scalers["x_scale"]
    mask = np.isfinite(y).astype(float)
    zy = np.where(mask.astype(bool), (y - scalers["y_mean"]) / scalers["y_scale"], 0.)
    return zx, zy, mask


def train_at_origin(panel, origin):
    x, y, indices, current = supervision(panel, origin)
    if not np.isfinite(current[:9]).all():
        return None
    # Validation forecast origins end 12 months before the current origin.
    # Calibration targets are censored before the first validation origin.
    validation_end = origin - max(HORIZONS)
    validation_start = validation_end - SETTINGS["validation_months"] + 1
    calibration_end = validation_start - 1
    cx, cy, _, _ = supervision(panel, calibration_end)
    select = (indices >= validation_start) & (indices <= validation_end)
    if not select.any():
        raise ValueError("No chronological validation observations")
    calibration_scalers = fit_scalers(cx, cy)
    tx, ty, tm = transform(cx, cy, calibration_scalers)
    vx, vy, vm = transform(x[select], y[select], calibration_scalers)
    selector = DenseMLP(seed=SETTINGS["seed"], l2=SETTINGS["l2"])
    chosen_epochs, selection_curve = selector.fit(
        tx, ty, tm, epochs=SETTINGS["max_epochs"], learning_rate=SETTINGS["learning_rate"],
        validation=(vx, vy, vm), patience=SETTINGS["patience"])
    scalers = fit_scalers(x, y)
    tx, ty, tm = transform(x, y, scalers)
    model = DenseMLP(seed=SETTINGS["seed"], l2=SETTINGS["l2"])
    _, curve = model.fit(tx, ty, tm, epochs=chosen_epochs, learning_rate=SETTINGS["learning_rate"])
    filled = np.where(np.isfinite(current), current, scalers["x_median"])
    prediction = model.forward(((filled - scalers["x_mean"]) / scalers["x_scale"])[None])[0][0]
    changes = prediction * scalers["y_scale"] + scalers["y_mean"]
    rates = changes.reshape(2, 9) + current[:9]
    return {
        "model": model, "scalers": scalers, "features": current, "predictions": rates,
        "training": {"origin_index": origin, "chosen_epochs": chosen_epochs,
                     "calibration_end_index": calibration_end,
                     "validation_start_index": validation_start,
                     "validation_end_index": validation_end,
                     "training_rows": len(x), "observed_targets": np.isfinite(y).sum(axis=0).tolist()},
        "curve": curve, "selection_curve": selection_curve,
    }
