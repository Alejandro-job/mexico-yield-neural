import json
from pathlib import Path
import sys
import tempfile
import hashlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import httpx
from huggingface_hub.errors import RepositoryNotFoundError
from safetensors.numpy import save_file
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from network import DenseMLP
from training import train_at_origin, supervision
from model import MexicanYieldForecaster
from yield_lab.data import load_rates
from upload_hf import check_artifacts, upload_model, ALLOW_FILES
from verify_model import validate_artifacts


class NeuralTests(unittest.TestCase):
    def test_backpropagation_matches_numerical_gradient(self):
        rng = np.random.default_rng(13)
        x, y = rng.normal(size=(5, 47)), rng.normal(size=(5, 18))
        mask = (rng.random((5, 18)) > .2).astype(float)
        model = DenseMLP(seed=7)
        _, gradient = model.loss_and_gradients(x, y, mask)
        for name, coordinate in [("W1", (3, 4)), ("W2", (1, 2)), ("W3", (2, 6)), ("b3", (7,))]:
            original = model.parameters[name][coordinate]
            epsilon = 1e-6
            model.parameters[name][coordinate] = original + epsilon
            plus = model.loss_and_gradients(x, y, mask)[0]
            model.parameters[name][coordinate] = original - epsilon
            minus = model.loss_and_gradients(x, y, mask)[0]
            model.parameters[name][coordinate] = original
            self.assertAlmostEqual((plus - minus) / (2 * epsilon), gradient[name][coordinate], places=6)

    def test_unobserved_targets_have_no_effect_on_loss(self):
        rng = np.random.default_rng(2)
        x, y = rng.normal(size=(4, 47)), rng.normal(size=(4, 18))
        mask = np.ones_like(y); mask[0, 3] = 0
        model = DenseMLP()
        expected = model.loss_and_gradients(x, y, mask)
        y[0, 3] = 1e10
        actual = model.loss_and_gradients(x, y, mask)
        self.assertEqual(expected[0], actual[0])
        for name in expected[1]:
            np.testing.assert_array_equal(expected[1][name], actual[1][name])

    def test_future_changes_cannot_affect_model_at_past_origin(self):
        _, panel, _ = load_rates(ROOT / "data/auction_rates.csv")
        with threadpool_limits(limits=1):
            first = train_at_origin(panel, 170)
            panel.iloc[171:] = 999.
            second = train_at_origin(panel, 170)
        np.testing.assert_array_equal(first["predictions"], second["predictions"])
        self.assertLess(first["training"]["calibration_end_index"],
                        first["training"]["validation_start_index"])
        self.assertLessEqual(first["training"]["validation_end_index"] + 12, 170)

    def test_targets_obey_calendar_horizons_and_availability(self):
        _, panel, _ = load_rates(ROOT / "data/auction_rates.csv")
        x, y, indices, _ = supervision(panel, 170)
        for row, origin in enumerate(indices):
            for k, h in enumerate([3, 12]):
                if origin + h > 170:
                    self.assertFalse(np.isfinite(y[row, k*9:(k+1)*9]).any())
                else:
                    expected = panel.iloc[origin+h].to_numpy(dtype=float) - x[row, :9]
                    np.testing.assert_allclose(y[row, k*9:(k+1)*9], expected, equal_nan=True)

    def test_model_export_predicts_both_horizons_and_rejects_wrong_shape(self):
        model = MexicanYieldForecaster(ROOT / "hf_model")
        sample = json.loads((ROOT / "hf_model/sample_input.json").read_text())
        actual = model.predict_features(sample["features"], sample["origin_month"])
        expected = json.loads((ROOT / "hf_model/prediction_example.json").read_text())
        self.assertEqual(actual["origin_month"], expected["origin_month"])
        for observed_part, expected_part in zip(actual["forecasts"], expected["forecasts"]):
            self.assertEqual(observed_part["horizon_months"], expected_part["horizon_months"])
            self.assertEqual([row["maturity"] for row in observed_part["rates"]],
                             [row["maturity"] for row in expected_part["rates"]])
            np.testing.assert_allclose([row["prediction_pct"] for row in observed_part["rates"]],
                                       [row["prediction_pct"] for row in expected_part["rates"]],
                                       rtol=1e-10, atol=1e-10)
        self.assertEqual([x["target_month"] for x in actual["forecasts"]], ["2026-12-01", "2027-09-01"])
        with self.assertRaises(ValueError):
            model.predict_features([7.] * 9)

    def test_neural_and_persistence_have_the_same_evaluation_pairs(self):
        results = pd.read_csv(ROOT / "outputs/comparison.csv")
        for horizon in (3, 12):
            subset = results[results.horizon_months == horizon].set_index("model")
            self.assertEqual(subset.loc["Red neuronal MLP", "observations"],
                             subset.loc["Persistencia", "observations"])
            self.assertEqual(subset.loc["Red neuronal MLP", "origins"],
                             subset.loc["Persistencia", "origins"])

    def test_upload_is_private_and_only_includes_the_model_allowlist(self):
        api = Mock()
        api.whoami.return_value = {"name": "fictional-test-owner"}
        api.repo_info.return_value = SimpleNamespace(private=True)
        api.upload_folder.return_value = SimpleNamespace(oid="a" * 40)
        api.list_repo_files.return_value = ALLOW_FILES
        receipt = upload_model(
            api, ROOT / "hf_model", "test-model",
            downloader=lambda **kwargs: ROOT / "hf_model/model.safetensors")
        self.assertEqual(receipt["visibility"], "private")
        self.assertEqual(api.upload_folder.call_args.kwargs["allow_patterns"], ALLOW_FILES)
        self.assertFalse(any(name.endswith(".csv") or ".env" in name for name in ALLOW_FILES))

    def test_existing_public_repository_is_not_modified(self):
        api = Mock()
        api.whoami.return_value = {"name": "fictional-test-owner"}
        api.repo_info.return_value = SimpleNamespace(private=False)
        with self.assertRaises(ValueError):
            upload_model(api, ROOT / "hf_model", "existing")
        api.upload_folder.assert_not_called()
        api.create_repo.assert_not_called()

    def test_new_repository_is_created_in_the_authenticated_users_account(self):
        api = Mock()
        api.whoami.return_value = {"name": "fictional-test-owner"}
        response = httpx.Response(404, request=httpx.Request(
            "GET", "https://huggingface.co/api/models/fictional-test-owner/new-model"))
        api.repo_info.side_effect = RepositoryNotFoundError(
            "Test-only missing repository", response=response)
        api.upload_folder.return_value = SimpleNamespace(oid="b" * 40)
        api.list_repo_files.return_value = ALLOW_FILES
        receipt = upload_model(
            api, ROOT / "hf_model", "new-model",
            downloader=lambda **kwargs: ROOT / "hf_model/model.safetensors")
        api.create_repo.assert_called_once_with(
            repo_id="fictional-test-owner/new-model", repo_type="model",
            visibility="private", exist_ok=False)
        self.assertEqual(receipt["repo_id"], "fictional-test-owner/new-model")

    def test_verifier_recalculates_metrics_and_forecasts(self):
        report = validate_artifacts(ROOT)
        self.assertEqual(report["parameters"], 1066)
        self.assertEqual(len(report["metrics"]), 12)
        self.assertEqual(sum(len(part["rates"]) for part in report["forecasts"]), 18)

    def test_model_rejects_wrong_tensor_shape_even_with_updated_checksum(self):
        original = MexicanYieldForecaster(ROOT / "hf_model")
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            tensors = dict(original.tensors)
            tensors["W1"] = tensors["W1"][:-1].copy()
            save_file(tensors, str(folder / "model.safetensors"))
            config = dict(original.config)
            config["weights_sha256"] = hashlib.sha256((folder / "model.safetensors").read_bytes()).hexdigest()
            (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid neural tensor: W1"):
                MexicanYieldForecaster(folder)
