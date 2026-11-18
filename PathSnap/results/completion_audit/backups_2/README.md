# PathSnap — Deliverable 1

Local distributed map matching and route reconstruction for the PES University ML hackathon. **Source implementation and local evidence only: not submitted or published.** No Git commands, uploads or Technical Report PDF were produced.

## Current status

All seven implementation stages run locally. Five models were trained with three-fold trip-grouped validation on the 80-trip training pool, then evaluated on all ten student-testing trips. Bayesian tuning completed ten trials. HMM decoding, a FastAPI backend, Streamlit UI, Folium maps and real HTTP integration are working. Reserved evaluator trips are not supplied; final evaluator testing and a physical two-laptop demonstration remain pending.

The current app is designed for whole-trip reconstruction. Labels used for pointwise evaluation are geometric/chronological reference-alignment **proxies**, not independently annotated per-GPS ground truth.

## Layout

```text
PathSnap/
  src/                  loaders, metric geometry, candidates, features, alignment
                        training, evaluation, tuning, HMM, inference, maps
  app/api.py            Laptop 1 FastAPI server
  ui/app.py             Laptop 2 Streamlit interface
  ui/client.py          shared HTTP client tested against the running API
  data/train/           80 extracted trip copies (not for publishing by default)
  data/student_test/    10 extracted trip copies
  data/evaluator_test/  empty; authorised final data only
  artifacts/            five baseline pipelines, tuned and deployed models
  artifacts/features/   local per-trip feature caches (excluded from publishing)
  configs/              features, seeds, folds, sample indices, deployment settings
  results/eda/          inventory, quality summaries, projection audit
  results/models/       comparisons, per-model JSON, graphs, confusion matrices
  results/tuning/       actual trial history, search results, tuned comparison
  results/hmm/          predictions, per-trip and aggregate comparisons
  results/maps/         interactive offline map (includes reference)
  results/logs/         actual execution, tests, HTTP and UI logs
  scripts/              extraction, launchers, audits, integration checks
  tests/                data, geometry, decoder, inference and API tests
  docs/methodology.md   assumptions, metrics and implementation details
  requirements.txt
```

## Installation

Use Python 3.11 (tested with 3.11.9). Open PowerShell in this folder. On a fresh machine:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --cache-dir .cache\pip -r requirements.txt
```

This machine already has a working `.venv` that reuses its installed scientific Python packages. New native GIS and SQL extensions were blocked by application control, so the implementation uses NumPy metre-based geometry and Optuna's in-memory storage with per-trial JSON persistence. Do not disable application-control policy. The app does not need pyproj or SQL storage.

Original input ZIP files remain in the parent folder. Extract reproducibly:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py
```

The extraction script verifies existing copies against archive content and refuses to overwrite differing data. The 360-file SHA-256 manifest is in `results/eda/source_manifest.json`.

Each trip directory is named with its eight-digit trip ID and contains matching `.track` (lon, lat, seconds), `.nodes` (lon, lat), `.arcs` (from-node, to-node), and `.route` (chronological arc IDs). All are whitespace-separated, headerless. IDs are zero-based row positions.

## Run now: Laptop 1

```powershell
cd 'D:\SEMESTERS\SEM5_2025\2.ML\Hackathon\PathSnap'
$env:PATHSNAP_MODE = 'student_test'
$env:PATHSNAP_HOST = '0.0.0.0'
$env:PATHSNAP_PORT = '8000'
.\.venv\Scripts\python.exe -m app.api
```

Run the Python commands above if local script execution is restricted. On a machine permitting local scripts, `.\scripts\run_backend.ps1 -Mode student_test -Port 8000` is a convenience wrapper.

Endpoints: `GET /health`, `GET /trips`, `POST /predict` with `{"trip_id":"<an ID from /trips>"}`. Interactive API documentation is at http://127.0.0.1:8000/docs . Keep the API terminal visible: it logs actual request paths, status, trip IDs, processing times and aggregate accuracy.

## Run now: Laptop 2 (or a second terminal locally)

Copy this project to Laptop 2 manually and install its dependencies. The UI only needs `ui/`, `src/__init__.py`, `src/config.py`, `src/visualizations.py` and installed packages; it does not need data or model artifacts.

```powershell
$env:PATHSNAP_API_URL = 'http://127.0.0.1:8000'
.\.venv\Scripts\python.exe -m streamlit run ui\app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false
```

Open http://localhost:8501 . Choose any of the ten actual student-testing trips and click **Predict selected trip**. Development mode is explicitly labelled. Returned metrics, GPS and predicted roads appear; ground-truth labels and reference geometry are never sent to the UI.

For two laptops, join the same Wi-Fi/hotspot. On Laptop 1 run `ipconfig` and identify its Wi-Fi IPv4 address. On Laptop 2 replace 127.0.0.1 with that address, e.g. `http://192.168.1.20:8000`, in PATHSNAP_API_URL or the sidebar. Do not enter 0.0.0.0 as the client address.

Verify from Laptop 2:

```powershell
Test-NetConnection -ComputerName 192.168.1.20 -Port 8000
Invoke-RestMethod http://192.168.1.20:8000/health
```

If needed, allow the backend Python executable/inbound TCP 8000 on the **Private** network profile using Windows Defender Firewall settings and the device owner's policy. No firewall changes were made by this implementation. Guest-network client isolation can prevent communication; try a shared hotspot. Cross-laptop verification is still pending.

For final evaluation, place the ten **authorised supplied** trip folders under `data/evaluator_test/` on Laptop 1 and restart with:

```powershell
$env:PATHSNAP_MODE = 'evaluator_test'
.\.venv\Scripts\python.exe -m app.api
```

No reserved trips or IDs are fabricated. An empty directory produces an empty list. Do not use this partition for training, feature selection or tuning. An alternative local data root can be set with PATHSNAP_DATA before server startup.

## Reproduce experiments

Preserve the existing successful artifacts, configs and results by copying this entire project to another local folder (omit .venv and .cache), or archiving those output folders before rerunning. Experiment scripts intentionally write their standard output paths. Run these steps sequentially; finish feature generation before training:

```powershell
.\.venv\Scripts\python.exe -m src.build_features
.\.venv\Scripts\python.exe scripts\audit_geometry.py
.\.venv\Scripts\python.exe -m src.train_models --folds 3 --max-observations 400
.\.venv\Scripts\python.exe -m src.tune_model --trials 10
.\.venv\Scripts\python.exe -m src.evaluate_hmm
.\.venv\Scripts\python.exe -m src.visualizations
.\.venv\Scripts\python.exe -m pytest tests -q
```

With the backend running, execute real socket and UI tests:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_http.py
.\.venv\Scripts\python.exe scripts\check_ui.py
```

No evaluator partition is read by the development experiment scripts. Feature caches contain labels for offline development, but production inference rebuilds features from track/nodes/arcs only. Learned scaling is embedded in relevant model pipelines. The selected tree model needs no fitted scaler; its feature order, deterministic preprocessing and decoder settings are saved in configs/deployment.json.

## Actual model results

Student testing includes all 19,160 observations. Macro F1 below is binary candidate-classification Macro F1. Top-1 is observation matching accuracy against aligned reference proxies. Inference times are classifier-only totals across the ten trips; they exclude geometry and HMM work.

| Model | CV Macro F1 | Student Top-1 | Student Macro F1 | Log loss | Inference s |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9030 | 83.66% | 0.9032 | 0.1158 | 0.0238 |
| Gaussian Naive Bayes | 0.8169 | 81.00% | 0.8159 | 0.4918 | 0.0491 |
| Decision Tree | 0.9047 | 84.67% | 0.9086 | 0.1331 | 0.0096 |
| Random Forest | 0.9115 | 84.61% | 0.9110 | 0.0974 | 0.9840 |
| HistGradientBoosting | 0.9134 | 85.28% | 0.9123 | 0.0949 | 0.8279 |

HistGradientBoosting was selected using CV matching accuracy and probability quality. Ten Optuna trials (three startup, seven TPE) took 95.88 seconds. Best parameters: max_iter=171, max_leaf_nodes=44, learning_rate=0.1193143151, l2_regularization=0.01779417046, min_samples_leaf=22. Seed=42; early_stopping=False.

The tuned CV Top-1 increased from 84.77% to 84.90%; the predefined objective increased from 0.84573 to 0.84706. Student Top-1 fell slightly to 85.25%, Macro F1 to 0.9120, and log loss worsened to 0.0957. Tuned classifier inference took 1.3026 seconds. The tuned variant is deployed according to the CV objective; tuning did **not** improve student-test results.

| Tuned-model route method | Top-1 | Direct route continuity | Mean sequence edit similarity |
|---|---:|---:|---:|
| Classifier only | 85.25% | 92.02% | 0.5816 |
| Classifier + HMM/Viterbi | 83.19% | 96.76% | 0.7160 |

HMM improves route structure but hurts pointwise matching. Per-trip results show both improvements and regressions. The saved all-trip inference run took 23.95 seconds total (concurrent system load affects timings). Candidate miss rate is 0.3967%. Do not confuse candidate accuracy (about 96%) with pointwise matching accuracy.

## Data findings and limitations

There are 208,417 training and 19,160 student-test observations. No missing/nonfinite GPS values, duplicate timestamps or reversed timestamps were found. There are 133 training gaps and two student-test gaps over 30 seconds. All networks have validated node/arc IDs. Full network connectivity and file-level statistics are saved.

Training reference alignment exceeds 80 m for 21,128 observations, concentrated especially in trips 00000054 and 00000033; those observations are excluded from fitting. This can reflect GPS/reference mismatch, incomplete route coverage or alignment limitations. The student set has 20 such observations, retained in evaluation. No claim is made that reference routes are error-free.

Maximum audited projected versus spherical segment-length difference is 1.14% on training networks and 0.49% on student-testing networks. See docs/methodology.md for exact alignment, sampling, feature, metric and HMM definitions.

The current decoder trades pointwise accuracy for continuity, has a 3000 m path-search cap and restarts at large gaps. Its probabilities are normalized classifier scores, not a separately calibrated generative observation model. A ten-trial search is a limited computational budget, not evidence of global optimality. Model families share the same approximate label construction.

Folium HTML is generated locally, but map tiles and JavaScript assets normally require Internet. Tile availability and rendering on the second laptop remain environment dependent. The UI has connection and API-error handling. If a port is occupied, change PATHSNAP_PORT/--server.port and the UI API address consistently. If model loading fails, check artifacts/best_model.joblib, configs/deployment.json and package versions. Load only trusted locally generated joblib files.

## Evidence and publishing checklist

- Real model metrics: results/models/comparison.csv and per-model JSON.
- Real tuning trials: results/tuning/trial_history.json and trials.csv.
- HMM results: results/hmm/per_trip.csv, summary.json and comparison.png.
- Folium offline map: results/maps/00000000_offline.html (contains permitted reference).
- Tests: results/logs/final_tests.log; 17 passed.
- Real HTTP requests: results/logs/http_smoke.json; all ten trips.
- Real UI button execution: results/logs/ui_smoke.json.
- Server request trace: results/logs/api_server.log.
- Earlier failed starts and their fixes remain in logs; they are not presented as successes.

Dependency verification passes (results/logs/dependencies.log). Browser-level visual inspection could not run because the Codex browser automation sandbox reports 'registered runtime ownership is missing'. Streamlit AppTest executed the actual UI and real HTTP prediction successfully; it does not verify browser tile/JavaScript rendering. That visual check remains pending.

All requested local components are implemented. Outstanding external checks: actual two-laptop network demonstration, reserved evaluator trips when supplied, and manual repository publishing. Before publishing, review .gitignore, confirm permission to share dataset-derived artifacts, retain required source/models/configs/results, omit .venv/.cache/raw data/feature caches, and follow the event's AI-assistance disclosure rules. This project was implemented with Codex assistance; the team must review and be able to explain the code and results. No submission or publication is claimed.


