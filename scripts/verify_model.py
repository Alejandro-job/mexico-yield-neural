"""Verify local artifacts and produce an offline HTML report; no network calls."""
import html
import io
import json
import platform
from pathlib import Path
import sys
from datetime import datetime, timezone
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from model import MexicanYieldForecaster
from yield_lab.data import load_rates, LABELS, causal_features
from yield_lab.features import ml_features


def validate_artifacts(root=ROOT):
    root = Path(root)
    model = MexicanYieldForecaster(root / "hf_model")
    config = model.config
    _, panel, source_hash = load_rates(root / "data/auction_rates.csv")
    if source_hash != config["source_csv_sha256"]:
        raise ValueError("El CSV no corresponde al modelo exportado")
    sample = json.loads((root / "hf_model/sample_input.json").read_text(encoding="utf-8"))
    if sample["origin_month"] != config["origin_month"]:
        raise ValueError("El ejemplo utiliza un mes distinto al del modelo")
    expected_features = ml_features(causal_features(panel)).iloc[-1].to_numpy(dtype=float)
    np.testing.assert_allclose(np.asarray(sample["features"], dtype=float),
                               expected_features, rtol=0, atol=1e-12, equal_nan=True)
    prediction = model.predict_features(sample["features"], sample["origin_month"])
    saved = json.loads((root / "hf_model/prediction_example.json").read_text(encoding="utf-8"))
    if [item["target_month"] for item in prediction["forecasts"]] != [
            item["target_month"] for item in saved["forecasts"]]:
        raise ValueError("El calendario del ejemplo no coincide")
    flatten = lambda result: [row["prediction_pct"] for part in result["forecasts"] for row in part["rates"]]
    np.testing.assert_allclose(flatten(prediction), flatten(saved), rtol=1e-10, atol=1e-10)
    if len(flatten(prediction)) != 18 or not np.isfinite(flatten(prediction)).all():
        raise ValueError("La inferencia no produjo 18 tasas finitas")
    parameters = sum(model.tensors[name].size for name in ("W1", "b1", "W2", "b2", "W3", "b3"))
    if parameters != 1066 or config["parameters"] != parameters:
        raise ValueError("El número de parámetros no coincide")

    neural = pd.read_csv(root / "outputs/neural_backtest.csv")
    baseline = pd.read_csv(root / "data/baseline_predictions.csv")
    provenance = json.loads((root / "data/baseline_metadata.json").read_text(encoding="utf-8"))
    if provenance["source_csv_sha256"] != source_hash:
        raise ValueError("Las referencias corresponden a otro CSV")
    keys = ["origin", "target", "horizon", "maturity"]
    if neural.duplicated(keys).any():
        raise ValueError("Hay pares duplicados en la evaluación neuronal")
    if set(neural.horizon.unique()) != {3, 12} or set(neural.maturity) != set(LABELS):
        raise ValueError("Horizontes o vencimientos inesperados en el backtest")
    if not np.isfinite(neural[["actual_pct", "prediction_pct", "error_bp"]]).all().all():
        raise ValueError("Hay valores no finitos en el backtest")
    for row in neural.itertuples():
        target = pd.Timestamp(row.origin) + pd.DateOffset(months=int(row.horizon))
        if target != pd.Timestamp(row.target):
            raise ValueError("Un objetivo no respeta su horizonte de calendario")
        actual = panel.loc[target].iloc[LABELS.index(row.maturity)]
        if not np.isfinite(actual) or not np.isclose(row.actual_pct, actual, rtol=0, atol=1e-10):
            raise ValueError("Un objetivo del backtest no coincide con el dato observado")
    baseline = baseline.merge(neural[keys], on=keys, how="inner", validate="many_to_one")
    persistence = baseline[baseline.model == "Persistencia"]
    if set(map(tuple, persistence[keys].to_numpy())) != set(map(tuple, neural[keys].to_numpy())):
        raise ValueError("Persistencia y red neuronal no usan los mismos pares")
    combined = pd.concat([neural, baseline], ignore_index=True)
    np.testing.assert_allclose(combined.error_bp,
                               (combined.prediction_pct - combined.actual_pct) * 100,
                               rtol=1e-9, atol=1e-9)
    metrics = []
    declared = pd.read_csv(root / "outputs/comparison.csv").set_index(["horizon_months", "model"])
    hf_metrics = pd.DataFrame(json.loads((root / "hf_model/metrics.json").read_text(encoding="utf-8"))).set_index(
        ["horizon_months", "model"])
    for (horizon, name), group in combined.groupby(["horizon", "model"]):
        item = {"horizon_months": int(horizon), "model": name,
                "rmse_bp": float(np.sqrt(np.mean(group.error_bp ** 2))),
                "mae_bp": float(np.mean(np.abs(group.error_bp))),
                "observations": len(group), "origins": int(group.origin.nunique())}
        for frame in (declared, hf_metrics):
            reference = frame.loc[(horizon, name)]
            np.testing.assert_allclose([item["rmse_bp"], item["mae_bp"]],
                                       [reference.rmse_bp, reference.mae_bp], rtol=1e-9, atol=1e-8)
            if item["observations"] != reference.observations or item["origins"] != reference.origins:
                raise ValueError("Los tamaños declarados no coinciden con las predicciones")
        metrics.append(item)
    training_file = root / "outputs/local_training_run.json"
    training = None
    if training_file.exists():
        candidate = json.loads(training_file.read_text(encoding="utf-8"))
        if (candidate.get("run_id") == config.get("training_run_id")
                and candidate.get("weights_sha256") == config["weights_sha256"]
                and candidate.get("source_csv_sha256") == source_hash):
            training = candidate
    return {"status": "passed", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "system": platform.system(), "python_version": platform.python_version(),
            "architecture": config["architecture"], "parameters": parameters,
            "weights_sha256": config["weights_sha256"], "origin_month": config["origin_month"],
            "forecasts": prediction["forecasts"], "metrics": metrics,
            "matching_training_record": training}


def write_report(report, root=ROOT):
    escape = lambda value: html.escape(str(value))
    keep = {"Red neuronal MLP", "Persistencia", "Ridge 9 tasas"}
    metric_rows = "".join(
        "<tr><td>" + "</td><td>".join(map(escape, [str(row["horizon_months"]) + " meses", row["model"],
            f'{row["rmse_bp"]:.2f}', f'{row["mae_bp"]:.2f}', row["observations"]])) + "</td></tr>"
        for row in sorted(report["metrics"], key=lambda row: (row["horizon_months"], row["rmse_bp"]))
        if row["model"] in keep)
    forecasts = "".join(
        "<tr><td>" + "</td><td>".join(map(escape, [part["target_month"], row["maturity"],
                                                f'{row["prediction_pct"]:.4f}%'])) + "</td></tr>"
        for part in report["forecasts"] for row in part["rates"])
    training = report["matching_training_record"]
    training_text = ("Hay un registro de entrenamiento que corresponde a estos pesos. Finalizó "
                     + escape(training["finished_at_utc"]) + " en " + escape(training["system"]) + "."
                     if training else "No hay un registro local de entrenamiento asociado a estos pesos. "
                     "Puedes estar usando el modelo entrenado incluido en el ZIP. Para crear un registro "
                     "ejecuta scripts/train_neural.py y vuelve a verificar.")
    template = '''<!doctype html><html lang="es"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Verificación de la red neuronal</title>
<style>body{font:16px/1.6 system-ui,sans-serif;color:#152641;background:#eef3f8;margin:0}
main{max-width:1050px;margin:auto;padding:40px 24px}section{background:white;padding:26px;margin:20px 0;border-radius:16px}
h1{font-size:34px;line-height:1.2}h2{font-size:23px}.ok{color:#06734b;font-weight:700}code{background:#edf2f7;padding:3px 7px;border-radius:5px}
table{width:100%;border-collapse:collapse}td,th{padding:10px;text-align:left;border-bottom:1px solid #d9e2ed}
th{background:#f3f7fa}.small{font-size:14px;color:#526680}img{width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}
.scroll{overflow:auto}</style><main><p class="small">MÉXICO · FORECASTING · COMPROBACIÓN LOCAL</p>
<h1>El modelo se carga y produce 18 pronósticos</h1><p class="ok">COMPROBACIONES COMPLETADAS</p>
<p>Verificado: __DATE__ · Sistema: __SYSTEM__ · Python: __PYTHON__</p>
<section><h2>Qué se verificó</h2><ul><li>Integridad SHA-256, dimensiones y valores finitos de los tensores.</li>
<li>1,066 parámetros y normalización válida.</li><li>Entradas reconstruidas desde el CSV y coincidencia con la inferencia guardada.</li>
<li>Calendarios, objetivos observados, pares comparables y métricas recalculadas.</li><li>__TESTS__ pruebas de implementación aprobadas, sin operaciones de red.</li></ul>
<p>Estas comprobaciones validan la implementación y los archivos. La capacidad de pronóstico se evalúa con el backtest de la tabla inferior.</p></section>
<section><h2>Entrenamiento en tu equipo</h2><p>__TRAINING__</p>
<p>Descomprimir el proyecto coloca los pesos en la carpeta; al ejecutar el entrenamiento se recalculan y se guarda un registro de esa ejecución.</p></section>
<section><h2>Arquitectura</h2><img src="../docs/red_neuronal.svg" alt="47 variables, capas ocultas de 16 y 8 neuronas y 18 salidas">
<p>Los pesos están en <code>hf_model/model.safetensors</code>. La estructura y la fecha de referencia están en <code>hf_model/config.json</code>.</p>
<pre>SHA-256: __SHA__</pre></section>
<section><h2>Calidad del pronóstico</h2><p>Errores en puntos base; menor es mejor. Un punto porcentual equivale a 100 puntos base.</p>
<div class="scroll"><table><tr><th>Horizonte</th><th>Modelo</th><th>RMSE</th><th>MAE</th><th>Pares</th></tr>__METRICS__</table></div>
<p>La red mejora frente a persistencia a tres meses y presenta más error a doce meses. Ridge obtiene menor error en ambos horizontes. Los datos son mensuales, las etiquetas se solapan y no se han calculado intervalos de predicción.</p></section>
<section><h2>Pronósticos generados</h2><p>Mes de referencia: __ORIGIN__. Tasas anualizadas; los valores son estimaciones.</p>
<div class="scroll"><table><tr><th>Mes objetivo</th><th>Vencimiento</th><th>Tasa pronosticada</th></tr>__FORECASTS__</table></div></section>
<section><h2>Validación temporal</h2><img src="../docs/validacion_temporal.svg" alt="Calibración, validación interna, reajuste y evaluación futura">
<p>La inferencia y las pruebas se ejecutaron en el sistema indicado arriba. Para comprobar tu PC, ejecuta nuevamente <code>python scripts/verify_model.py</code>.</p></section></main></html>'''
    values = {"__DATE__": escape(report["checked_at_utc"]), "__SYSTEM__": escape(report["system"]),
              "__PYTHON__": escape(report["python_version"]), "__TESTS__": str(report["tests_passed"]),
              "__TRAINING__": training_text, "__SHA__": escape(report["weights_sha256"]),
              "__METRICS__": metric_rows, "__ORIGIN__": escape(report["origin_month"]),
              "__FORECASTS__": forecasts}
    for key, value in values.items():
        template = template.replace(key, value)
    output = Path(root) / "outputs"
    output.mkdir(exist_ok=True)
    (output / "verification_report.html").write_text(template, encoding="utf-8")
    (output / "local_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    report = validate_artifacts()
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("Fallaron las pruebas:\n" + stream.getvalue())
    report["tests_passed"] = result.testsRun
    write_report(report)
    print("[OK] Pesos, dimensiones y 1,066 parámetros válidos")
    print("[OK] 18 pronósticos reproducidos y métricas recalculadas")
    print("[OK] " + str(result.testsRun) + " pruebas aprobadas; sin llamadas de red")
    print("Sistema comprobado: " + report["system"] + " | Python " + report["python_version"])
    print("Registro de entrenamiento asociado: " + ("SÍ" if report["matching_training_record"] else "NO"))
    print("Informe: outputs/verification_report.html")
    print("VERIFICACION COMPLETADA")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("[ERROR] " + str(exc), file=sys.stderr)
        sys.exit(1)
