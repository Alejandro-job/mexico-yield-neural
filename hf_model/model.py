"""Portable NumPy inference for the neural model stored on Hugging Face."""
import json
from pathlib import Path
import hashlib
from datetime import date
import numpy as np
from safetensors.numpy import load_file


class MexicanYieldForecaster:
    def __init__(self, directory):
        directory = Path(directory)
        self.config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        weights = directory / "model.safetensors"
        if hashlib.sha256(weights.read_bytes()).hexdigest() != self.config["weights_sha256"]:
            raise ValueError("The neural weights do not match config.json")
        self.tensors = load_file(weights)
        if self.config["architecture"] != [47, 16, 8, 18]:
            raise ValueError("Unsupported neural architecture")
        shapes = {"W1": (47, 16), "b1": (16,), "W2": (16, 8), "b2": (8,),
                  "W3": (8, 18), "b3": (18,), "x_median": (47,), "x_mean": (47,),
                  "x_scale": (47,), "y_mean": (18,), "y_scale": (18,)}
        if set(self.tensors) != set(shapes):
            raise ValueError("Unexpected neural tensor names")
        for name, shape in shapes.items():
            tensor = self.tensors[name]
            if tensor.shape != shape or not np.isfinite(tensor).all():
                raise ValueError("Invalid neural tensor: " + name)
        if (self.tensors["x_scale"] <= 0).any() or (self.tensors["y_scale"] <= 0).any():
            raise ValueError("Normalization scales must be positive")
        if self.config["horizons_months"] != [3, 12] or len(self.config["maturities"]) != 9:
            raise ValueError("Unexpected forecast horizons or maturities")

    @classmethod
    def from_pretrained(cls, repo_id, revision=None, token=None):
        from huggingface_hub import snapshot_download
        directory = snapshot_download(
            repo_id=repo_id, revision=revision, token=token,
            allow_patterns=["config.json", "model.safetensors"])
        return cls(directory)

    def predict_features(self, features, origin_month=None):
        x = np.asarray(features, dtype=float)
        if x.shape != (47,) or not np.isfinite(x[:9]).all() or np.isinf(x).any():
            raise ValueError("Provide 47 features with nine finite current yields; use null for missing lags")
        t = self.tensors
        filled = np.where(np.isfinite(x), x, t["x_median"])
        z = (filled - t["x_mean"]) / t["x_scale"]
        hidden1 = np.maximum(z @ t["W1"] + t["b1"], 0)
        hidden2 = np.maximum(hidden1 @ t["W2"] + t["b2"], 0)
        normalized = hidden2 @ t["W3"] + t["b3"]
        changes = normalized * t["y_scale"] + t["y_mean"]
        rates = changes.reshape(2, 9) + x[:9]
        reference = date.fromisoformat(origin_month or self.config["origin_month"])
        if reference.day != 1:
            raise ValueError("origin_month must be the first day of the reference month")
        output = {"origin_month": reference.isoformat(), "forecasts": []}
        for k, horizon in enumerate(self.config["horizons_months"]):
            shifted = reference.year * 12 + reference.month - 1 + horizon
            target = date(shifted // 12, shifted % 12 + 1, 1).isoformat()
            output["forecasts"].append({
                "horizon_months": horizon,
                "target_month": target,
                "rates": [{"maturity": maturity, "prediction_pct": float(rates[k, j])}
                          for j, maturity in enumerate(self.config["maturities"])]})
        return output
