from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

INSTRUMENTS = [
    "CETEs, 28 days", "CETEs, 91 days", "CETEs, 182 days",
    "CETEs, 364 days", "MBONOs, 3 years", "MBONOs, 5 years",
    "MBONOs, 10 years", "MBONOs, 20 years", "MBONOs, 30 years",
]
LABELS = ["28d", "91d", "182d", "364d", "3a", "5a", "10a", "20a", "30a"]
MODELS = ["Persistencia", "Univariado", "Ridge 9 tasas", "PCA 3 + ridge", "IA · Gradient boosting"]


def load_rates(path):
    path = Path(path)
    raw = pd.read_csv(path)
    required = {"DATE", "INSTRUMENT", "INDICATOR", "VALUE_PCT"}
    if set(raw.columns) != required:
        raise ValueError("Unexpected CSV columns")
    raw["DATE"] = pd.to_datetime(raw["DATE"], errors="raise")
    raw["VALUE_PCT"] = pd.to_numeric(raw["VALUE_PCT"], errors="raise")
    if raw.duplicated(["DATE", "INSTRUMENT"]).any():
        raise ValueError("Duplicate instrument-month")
    if set(raw.INSTRUMENT) != set(INSTRUMENTS):
        raise ValueError("Unexpected instrument set")
    if not np.isfinite(raw.VALUE_PCT).all():
        raise ValueError("Non-finite observed yield")
    if not raw.DATE.dt.is_month_start.all():
        raise ValueError("DATE must label the first day of the month")
    idx = pd.date_range(raw.DATE.min(), raw.DATE.max(), freq="MS")
    panel = raw.pivot(index="DATE", columns="INSTRUMENT", values="VALUE_PCT")
    panel = panel.reindex(index=idx, columns=INSTRUMENTS)
    return raw, panel, hashlib.sha256(path.read_bytes()).hexdigest()


def causal_features(panel, limit=2):
    """Carry an observed rate forward at most limit calendar months."""
    return panel.ffill(limit=limit)


def pca_summary(frame):
    values = frame.dropna().to_numpy(dtype=float)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale == 0] = 1
    z = (values - mean) / scale
    _, singular, vectors = np.linalg.svd(z, full_matrices=False)
    shares = singular ** 2 / np.sum(singular ** 2)
    # Signs are arbitrary; make the largest absolute coefficient positive.
    for vector in vectors:
        if vector[np.argmax(np.abs(vector))] < 0:
            vector *= -1
    return {
        "rows": len(values), "shares": shares.tolist(),
        "cumulative": np.cumsum(shares).tolist(),
        "loadings": vectors[:3].tolist(),
    }
