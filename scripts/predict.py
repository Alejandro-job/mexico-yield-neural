"""Run the exported local model or download its tensors from your Hub repo."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from model import MexicanYieldForecaster

parser = argparse.ArgumentParser()
parser.add_argument("--repo", help="Optional Hugging Face model repository ID")
parser.add_argument("--revision", help="Optional exact Hub commit for reproducible loading")
source = parser.add_mutually_exclusive_group()
source.add_argument("--input", type=Path, help="JSON with 47 features and origin_month")
source.add_argument("--csv", type=Path, help="Monthly CSV with the same schema as data/auction_rates.csv")
args = parser.parse_args()
if args.csv:
    import numpy as np
    from yield_lab.data import load_rates, causal_features
    from yield_lab.features import ml_features
    _, panel, _ = load_rates(args.csv)
    if len(panel) < 4:
        parser.error("Provide at least four calendar months to construct the lagged features")
    features = ml_features(causal_features(panel)).iloc[-1].to_numpy(dtype=float)
    payload = {"origin_month": panel.index[-1].strftime("%Y-%m-%d"),
               "features": [float(value) if np.isfinite(value) else None for value in features]}
else:
    path = args.input or ROOT / "hf_model/sample_input.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
model = (MexicanYieldForecaster.from_pretrained(args.repo, revision=args.revision)
         if args.repo else MexicanYieldForecaster(ROOT / "hf_model"))
print(json.dumps(model.predict_features(payload["features"], payload["origin_month"]),
                 ensure_ascii=False, indent=2))
