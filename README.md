# Révise Malin

Révise Malin ranks the course notions a student is most likely to fail next, so they revise those first. It combines a graph recommender (NLGCL, RecSys 2025) with a learned time term, trained on **DBE-KT22**, the answers of Database Systems students of the Australian National University to 212 exercises tagged with 93 knowledge components ("Projection", "Normalisation", "Transactions"...).

Team: Grégoire Champourlier, Maxence Simonet, Hippolyte Nussbaumer, Dan Waintraub (ECE Paris, ING5 Data & AI).

**Status (Week 4, showcase):** the trained model is served end to end. The Streamlit interface calls the FastAPI backend, the API ranks notions with the trained model (no mocks, about 1 ms per request), and feedback clicks are stored in SQLite.

**Dataset change.** Weeks 2 and 3 used EdNet (TOEIC, 5,000 students). For the showcase we moved to DBE-KT22: its knowledge components have names and descriptions, so the recommendations can be read and explained, and the whole pipeline runs in a few minutes instead of hours. The EdNet results of Week 3 are kept in `reports/ednet_week3/` and `artifacts/ednet_week3/`.

## Run the system locally (about 5 minutes, Python 3.11 to 3.14)

The trained model ships with the repository (`artifacts/serving.npz`, 0.4 MB), so you need neither the dataset nor PyTorch to run the demo.

1. Clone the repository
   ```bash
   git clone https://github.com/ApparentlyGregz/RecommenderSystemProject.git
   cd RecommenderSystemProject
   ```
2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
   ```
3. Install the dependencies
   ```bash
   pip install -r requirements.txt
   ```
4. Start the API and the interface
   ```bash
   python run.py
   ```
5. Open **http://localhost:8501** for the interface, and http://localhost:8000/docs for the API.

`python run.py --lan` listens on every network interface, so another laptop on the same WiFi can open `http://<your IP>:8501`. `python run.py --api-only` starts only the API. Ports can be changed with `--port` (API) and `--ui-port` (interface).

### What you can do in the interface

- Pick a student (or 🎲 a random one) among the 718 DBE-KT22 students of the sample: answers, error rate, error rate per question difficulty, weakest notions so far.
- See the notions they should revise first. Each one shows its score split into the **graph part** (students who fail the same notions fail this one too) and the **time part** (learned from when the notion was last practised), the student's errors on it, a sentence that explains the ranking from those numbers, and, under "What is it?", the DBE-KT22 description and related notions.
- Move the **"Revise in … days"** slider: the time term is recomputed for the shifted dates.
- Click **Revised**, **Already known** or **Not useful**: the click goes to `POST /log_interaction`, is stored in `data/feedback.db`, the notion leaves the list, and the stored clicks are shown under the recommendations ("Reset my feedback" brings them back).

### API

| Endpoint | What it does |
|---|---|
| `GET /recommend?student_id=0&k=10&days_ahead=0&practised_only=false` | Top-k notions from the trained model, with score parts, evidence and explanation (`"mode": "model"`) |
| `GET /students/{id}` | History summary of a student |
| `POST /log_interaction` | `{"student_id": 0, "topic": 66, "action": "revised" \| "already_known" \| "not_useful"}` stored in SQLite |
| `GET /interactions/{id}`, `DELETE /interactions/{id}` | List or reset a student's clicks |
| `GET /health` | Status and the loaded model |

## Results (test cut, 703 students)

From `reports/test_results.md` (paired bootstrap over students, 1,000 resamples). Each model ranks only the notions the student attempted in the test window; relevant = failed at least once.

| Model | NDCG@10 | Recall@5 | Recall@10, notions not practised for > 14 days |
|---|---|---|---|
| Random | 0.601 | 0.313 | 0.642 |
| Most-popular (attempt frequency) | 0.749 | 0.414 | 0.959 |
| Global error rate | 0.723 | 0.412 | 0.447 |
| Personal error rate | 0.716 | 0.408 | 0.526 |
| **Full model (NLGCL + time term), served** | **0.805** | 0.467 | 0.509 |
| NLGCL, no time term | 0.800 | 0.450 | 0.770 |
| LightGCN + time term | 0.799 | **0.468** | 0.460 |
| LightGCN | 0.801 | 0.453 | 0.770 |

What we take from it:

- The full model beats every baseline on NDCG@10 (+0.056 over most-popular, 95% interval [+0.044, +0.068]) and on Recall@5. It is the one chosen on validation (NDCG@10 0.822) and the one the API serves.
- Recall@20 is about 0.99 for every method: a student attempts a median of 17 notions per 25-answer window, so a top 20 almost always contains all of them. We report it in the full table but do not use it to compare models.
- The time term does **not** behave like forgetting on this dataset. The learned β is negative (−0.025): notions practised in the last day are the riskiest, because a student working on a chapter answers its new questions next. That is why the variants with the time term rank old notions lower (Recall@10 on old notions 0.51 against 0.77). The interface says so in its explanations instead of showing a forgetting curve the data does not support. The same effect appeared on EdNet in Week 3.
- Gains over the ablations are small: NLGCL adds +0.005 NDCG@10 over LightGCN with the same time term (significant), and the full model is +0.004 over NLGCL alone (not significant).
- Personalisation is partial: across students, 60 of the 93 notions appear in some top 10 and 22 different notions come first, but hard notions such as Projection and Relational Algebra appear in most lists.

## Rebuild the data and the model (about 5 minutes on a laptop CPU)

```bash
pip install -r requirements-train.txt
```

1. `python scripts/download_dbe_kt22.py` downloads the DBE-KT22 CSV files (17 MB) into `data/raw/dbe-kt22/`.
2. `python -m src.prepare_data` keeps the students with at least 125 answers and writes `data/processed/` (events, topics, stats).
3. `python -m src.train` trains the full model (about 30 s) and saves `artifacts/model_full.pt`. Ablations: `--no-forgetting` (no time term), `--no-contrastive` (LightGCN), or both.
4. `python -m src.evaluate_test` scores every `artifacts/model_*.pt` and the baselines on the test cut, with paired bootstrap intervals, and writes `reports/test_results.md` and `.json`.
5. `python -m src.export_serving` propagates the trained model on each student's full history and writes `artifacts/serving.npz`, the file the API loads. It checks that the numpy predictor gives the same scores as the PyTorch model.
6. `pytest` runs the tests (split without leakage, metrics, model, serving predictor, API).

The scripts `check_week2.py`, `sample_users.py`, `build_week3_report.py` and `make_synthetic.py` produced the EdNet evidence of Weeks 2 and 3 and are kept for the record; they expect the EdNet format.

## How it works

- **Data** (`src/prepare_data.py`). One row per (answer, knowledge component): a question tagged with two KCs gives two rows. 718 of the 1,264 students have at least 125 answers (136,130 answers, error rate 23%).
- **Split without leakage.** Each student's answers are sorted by time. Three label windows of 25 answers (train, validation, test) sit at the end of the history, with at least 50 answers before the first one. For each window, inputs are everything strictly before it. Cuts follow timestamps, so answers given at the same moment are never split.
- **Labels.** A notion attempted and failed at least once in the window is positive; attempted and always correct is negative.
- **Model** (`src/model.py`). LightGCN on a student-notion graph (edge = past error), the NLGCL neighbour-layer contrastive loss, and a time term in two places: edges weighted by 2^(-Δt/h) and a score term β(1 - 2^(-Δt/h)), with one learned half-life h per notion and a learned value for notions never practised.
- **Serving** (`src/export_serving.py`, `src/predict.py`). At export time the learned embeddings are propagated once on the graph of each student's full history. The API keeps them fixed and recomputes the time term at every request. It uses numpy only.
- **Evaluation** (`src/evaluate.py`, `src/evaluate_test.py`). Only notions the student attempted in the window are ranked. Metrics: NDCG@10, Recall@20, Recall@5, and Recall@10 on failed notions not practised for more than 14 days. Baselines: random, most-popular, global error rate, personal error rate, recent activity.

## Project structure

```
src/        config, prepare_data, build_graph, model, train, evaluate, evaluate_test, export_serving, predict
api/        FastAPI app and SQLite storage
app/        Streamlit interface (talks to the API only)
artifacts/  serving.npz (served model), training histories and logs; ednet_week3/ for the Week 3 runs
reports/    test results; ednet_week3/ and the Week 3 report for the EdNet runs
scripts/    DBE-KT22 download, Week 2-3 evidence scripts (EdNet)
sim/        toy simulations behind the design choices
tests/      pytest suite
```

## Data licence and references

DBE-KT22 is published by the Australian National University on the Australian Data Archive ([doi:10.26193/6DZWOH](https://doi.org/10.26193/6DZWOH)), free to use with citation: Abdelrahman et al., *DBE-KT22: A Knowledge Tracing Dataset Based on Online Student Evaluation*, 2022 ([arXiv:2208.12651](https://arxiv.org/abs/2208.12651)). The download script uses a public mirror of the same files on [Hugging Face](https://huggingface.co/datasets/Unggi/dbe-kt22_raw_data). We use it for coursework only.

- Xu et al., *NLGCL: Naturally Existing Neighbor Layers Graph Contrastive Learning for Recommendation*, RecSys 2025 ([code](https://github.com/Jinfeng-Xu/NLGCL)).
- He et al., *LightGCN*, SIGIR 2020.
- Settles & Meeder, *A Trainable Spaced Repetition Model for Language Learning*, ACL 2016.
- Choi et al., *EdNet: A Large-Scale Hierarchical Dataset in Education*, 2020 (Weeks 2-3, CC BY-NC 4.0).
