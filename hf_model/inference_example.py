import json
from pathlib import Path
from model import MexicanYieldForecaster

folder = Path(__file__).resolve().parent
payload = json.loads((folder / "sample_input.json").read_text(encoding="utf-8"))
model = MexicanYieldForecaster(folder)
print(json.dumps(model.predict_features(payload["features"], payload["origin_month"]),
                 ensure_ascii=False, indent=2))
