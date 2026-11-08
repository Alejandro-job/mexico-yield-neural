"""Train, evaluate and export a real MLP, without uploading to any service."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import platform
import uuid
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from safetensors.numpy import save_file
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from yield_lab.data import load_rates, LABELS
from yield_lab.features import ML_FEATURE_NAMES
from training import train_at_origin, HORIZONS
from network import ARCHITECTURE, SETTINGS

NAME = "Red neuronal MLP"


def records(frame):
    return json.loads(frame.to_json(orient="records"))


def main():
    started_at = datetime.now(timezone.utc).isoformat()
    run_id = uuid.uuid4().hex
    _, panel, source_hash = load_rates(ROOT / "data/auction_rates.csv")
    provenance = json.loads((ROOT / "data/baseline_metadata.json").read_text(encoding="utf-8"))
    if provenance["source_csv_sha256"] != source_hash:
        raise ValueError("The baseline snapshot belongs to a different CSV. Recompute "
                         "baseline_predictions.csv and baseline_metadata.json before evaluating new data.")
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    neural = []
    with threadpool_limits(limits=1):
        for origin in range(119, len(panel) - 3):
            trained = train_at_origin(panel, origin)
            if trained is None:
                continue
            for k, h in enumerate(HORIZONS):
                if origin + h >= len(panel):
                    continue
                actual = panel.iloc[origin + h].to_numpy(dtype=float)
                for j, maturity in enumerate(LABELS):
                    if not np.isfinite(actual[j]):
                        continue
                    predicted = float(trained["predictions"][k, j])
                    neural.append({
                        "origin": panel.index[origin].strftime("%Y-%m-%d"),
                        "target": panel.index[origin + h].strftime("%Y-%m-%d"),
                        "horizon": h, "maturity": maturity, "model": NAME,
                        "actual_pct": actual[j], "prediction_pct": predicted,
                        "error_bp": (predicted - actual[j]) * 100,
                        "chosen_epochs": trained["training"]["chosen_epochs"]})
            if (origin - 119) % 30 == 0:
                print("Evaluated origin", panel.index[origin].strftime("%Y-%m"), flush=True)
        final = train_at_origin(panel, len(panel) - 1)
    nn = pd.DataFrame(neural)
    baseline = pd.read_csv(ROOT / "data/baseline_predictions.csv")
    keys = ["origin", "target", "horizon", "maturity"]
    common = nn[keys].drop_duplicates()
    baseline = baseline.merge(common, on=keys, how="inner")
    truth = nn[keys + ["actual_pct"]].rename(columns={"actual_pct": "neural_actual_pct"})
    checked = baseline.merge(truth, on=keys, how="left", validate="many_to_one")
    if not np.allclose(checked.actual_pct, checked.neural_actual_pct, rtol=0, atol=1e-10):
        raise AssertionError("Baseline and neural targets differ")
    expected = baseline[baseline.model == "Persistencia"][keys]
    if set(map(tuple, common.to_numpy())) != set(map(tuple, expected.to_numpy())):
        raise AssertionError("Neural and baseline evaluation pairs differ")
    combined = pd.concat([baseline, nn], ignore_index=True)
    metrics = []
    for (h, model), group in combined.groupby(["horizon", "model"]):
        metrics.append({
            "horizon_months": int(h), "model": model,
            "rmse_bp": float(np.sqrt(np.mean(group.error_bp ** 2))),
            "mae_bp": float(np.mean(np.abs(group.error_bp))),
            "observations": len(group), "origins": int(group.origin.nunique())})
    metrics = pd.DataFrame(metrics).sort_values(["horizon_months", "rmse_bp"])
    nn.to_csv(output / "neural_backtest.csv", index=False)
    metrics.to_csv(output / "comparison.csv", index=False)
    hf = ROOT / "hf_model"
    hf.mkdir(exist_ok=True)
    tensors = {**final["model"].parameters, **final["scalers"]}
    save_file({key: np.ascontiguousarray(value) for key, value in tensors.items()},
              str(hf / "model.safetensors"), metadata={"architecture": "47-16-8-18", "framework": "numpy"})
    config = {
        "architecture": list(ARCHITECTURE), "activation": "relu",
        "output": "standardized future yield changes; add current yields after inverse scaling",
        "parameters": sum(value.size for value in final["model"].parameters.values()),
        "feature_names": ML_FEATURE_NAMES, "maturities": LABELS, "horizons_months": list(HORIZONS),
        "origin_month": panel.index[-1].strftime("%Y-%m-%d"), "settings": SETTINGS,
        "training": final["training"], "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_run_id": run_id,
        "source_csv_sha256": source_hash,
        "weights_sha256": hashlib.sha256((hf / "model.safetensors").read_bytes()).hexdigest(),
    }
    (hf / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (hf / "metrics.json").write_text(json.dumps(records(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    sample = {"origin_month": config["origin_month"],
              "features": [float(x) if np.isfinite(x) else None for x in final["features"]]}
    (hf / "sample_input.json").write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(ROOT / "src/model.py", hf / "model.py")
    from model import MexicanYieldForecaster
    reloaded = MexicanYieldForecaster(hf)
    prediction = reloaded.predict_features(sample["features"], sample["origin_month"])
    expected_rates = np.array([[x["prediction_pct"] for x in row["rates"]] for row in prediction["forecasts"]])
    np.testing.assert_allclose(expected_rates, final["predictions"], rtol=1e-12, atol=1e-12)
    (hf / "prediction_example.json").write_text(json.dumps(prediction, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "training_curve.json").write_text(json.dumps(final["curve"], indent=2), encoding="utf-8")
    (output / "epoch_selection_curve.json").write_text(json.dumps(final["selection_curve"], indent=2), encoding="utf-8")
    receipt = {"run_id": run_id, "started_at_utc": started_at,
               "finished_at_utc": datetime.now(timezone.utc).isoformat(),
               "system": platform.system(), "python_version": platform.python_version(),
               "origin_month": config["origin_month"],
               "parameters": config["parameters"], "chosen_epochs": final["training"]["chosen_epochs"],
               "source_csv_sha256": source_hash, "weights_sha256": config["weights_sha256"]}
    (output / "local_training_run.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False), flush=True)
    print("Architecture", config["architecture"], "Parameters", config["parameters"],
          "Chosen epochs", config["training"]["chosen_epochs"], flush=True)
    print("TRAINING COMPLETED | hf_model/model.safetensors | outputs/local_training_run.json", flush=True)


if __name__ == "__main__":
    main()
