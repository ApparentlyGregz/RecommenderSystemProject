# Révise Malin

Révise Malin ranks the course notions a student is most likely to fail next, so they revise those first. It combines a graph recommender (NLGCL, RecSys 2025) with a learned time term, trained on **DBE-KT22**, the answers of Database Systems students of the Australian National University to 212 exercises tagged with 93 knowledge components ("Projection", "Normalisation", "Transactions"...).

Team: Grégoire Champourlier, Maxence Simonet, Hippolyte Nussbaumer, Dan Waintraub, Martin Mialaret (ECE Paris, ING5 Data & AI).

**Status (Week 4, showcase):** the trained model is served end to end. The Streamlit interface calls the FastAPI backend, the API ranks notions with the trained model (no mocks, about 1 ms per request), students practise on real DBE-KT22 questions whose answers move the ranking, and everything they do is stored in SQLite.

**Dataset change.** Weeks 2 and 3 used EdNet (TOEIC, 5,000 students). For the showcase we moved to DBE-KT22: its knowledge components have names and descriptions, so the recommendations can be read and explained, and the whole pipeline runs in a few minutes instead of hours. The EdNet results of Week 3 are kept in `reports/ednet_week3/` and `artifacts/ednet_week3/`.

## Run the system locally (about 5 minutes, Python 3.11 to 3.14)

The trained model ships with the repository (`artifacts/serving.npz`, 0.4 MB), so you need neither PyTorch nor training. On first launch `run.py` downloads the DBE-KT22 files (17 MB) to build the practice questions, whose text is not in Git; without network the app runs without the practice mode.

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
- Click **📝 Practise this notion** to answer a real DBE-KT22 question on it (relational algebra shown as formulas). The correction appears above the list, and the notion moves: up after a wrong answer, down after a correct one (the model's percentile counts as two answers, then each answer in the app moves it). Questions that need tables from the course worksheet, which the dataset lacks, come last with a note.
- Move the **"Revise in … days"** slider: the time term is recomputed for the shifted dates.
- Click **Revised**, **Already known** or **Not useful**: the click goes to `POST /log_interaction`, is stored in `data/feedback.db`, the notion leaves the list, and the stored clicks are shown under the recommendations ("Reset my feedback" brings them back).

### API

| Endpoint | What it does |
|---|---|
| `GET /recommend?student_id=0&k=10&days_ahead=0&practised_only=false` | Top-k notions from the trained model, with score parts, evidence and explanation (`"mode": "model"`) |
| `GET /students/{id}` | History summary of a student |
| `GET /questions?student_id=0&topic=66&n=1` | Real DBE-KT22 questions of a notion, without their answers |
| `POST /answer` | `{"student_id": 0, "question": 146, "choice": 601}`: checks, stores in SQLite, returns the correction; the notion moves in the next `/recommend` |
| `DELETE /answers/{id}` | Reset a student's practice answers |
| `POST /log_interaction` | `{"student_id": 0, "topic": 66, "action": "revised" \| "already_known" \| "not_useful"}` stored in SQLite |
| `GET /interactions/{id}`, `DELETE /interactions/{id}` | List or reset a student's clicks |
| `GET /health` | Status and the loaded model |

## Results (test cut, 710 students)

Each model ranks only the notions the student attempted in the test window. **A notion is relevant if the student's error rate on it is above their own error rate in that window** ("weaker than usual"). The Week 3 report showed that the older label, "failed at least once", grows with how often a notion is shown and so rewards the most-practised notions; it is kept below as a secondary result. Full tables: `reports/test_results.md` (paired bootstrap, 1,000 resamples), `reports/test_results_any_fail.md` and `reports/seed_summary.md` (three training seeds).

| Model | NDCG@10 | Recall@5 | Recall@10, notions not practised for > 14 days |
|---|---|---|---|
| Random | 0.528 | 0.315 | 0.663 |
| Most-popular (attempt frequency) | 0.580 | 0.360 | 0.941 |
| Personal error rate | 0.662 | 0.443 | 0.519 |
| Global error rate (notion difficulty) | 0.668 | **0.447** | 0.448 |
| **Full model (NLGCL + time term), served** | **0.686** ± 0.001 | 0.422 ± 0.001 | 0.690 |
| LightGCN + time term | 0.686 ± 0.001 | 0.423 ± 0.002 | 0.668 |
| NLGCL, no time term | 0.683 ± 0.003 | 0.417 ± 0.001 | 0.789 |
| LightGCN | 0.685 ± 0.001 | 0.417 ± 0.002 | 0.786 |

(± = standard deviation over three training seeds; Recall@10 on old notions is from seed 42, 97 students.)

What we take from it:

- The full model beats every baseline on NDCG@10, including the notion difficulty (+0.019, 95% interval [+0.008, +0.030]) and most-popular (+0.107). The difficulty baseline stays ahead on Recall@5.
- **With the corrected label the time term is learned the right way round**: β = +0.049 (median half-life 6.9 days), so a notion gets riskier the longer it has not been practised. With "failed at least once" β came out negative (-0.025): recency then meant "the chapter the student is drilling now". The time term adds +0.010 validation NDCG@10 (0.723 vs 0.713, three seeds); on test the four variants are within 0.003.
- **The time term does not help on old notions.** On failed notions not practised for more than 14 days, Recall@10 drops from 0.789 (NLGCL, no time term) to 0.690 with it (gap -0.099, 95% interval [-0.155, -0.051], seed 42). The learned value for a never-practised notion (+0.20) is four times the largest forgetting effect (beta = +0.049), so the time term mostly ranks new notions first. Balancing the two is the next step.
- Every model has a learned difficulty per notion (a topic bias: +0.006 validation NDCG@10 for the full model, 0.723 vs 0.716 over three seeds). The contrastive loss adds nothing measurable here.
- Recall@20 is about 0.99 for every method (median 17 candidates per window), so it is reported but not used to compare models.
- With the old label (`reports/test_results_any_fail.md`) the full model reaches 0.781 NDCG@10 against 0.749 for most-popular.

## Rebuild the data and the model (about 5 minutes on a laptop CPU)

```bash
pip install -r requirements-train.txt
```

1. `python scripts/download_dbe_kt22.py` downloads the DBE-KT22 CSV files (17 MB) into `data/raw/dbe-kt22/`.
2. `python -m src.prepare_data` keeps the students with at least 125 answers and writes `data/processed/` (events, topics, stats). `python -m src.questions` builds the practice questions (`data/processed/questions.parquet`, not in Git).
3. `python -m src.train` trains the full model (about 30 s) and saves `artifacts/model_full.pt`. Ablations: `--no-forgetting` (no time term), `--no-contrastive` (LightGCN), or both; `--seed 1 --tag full_s1` for more seeds, then `python scripts/seed_summary.py`. The learning rate, batch size and topic bias were checked on validation only.
4. `python -m src.evaluate_test --models artifacts/model_full.pt artifacts/model_cl_noforget.pt artifacts/model_nocl_forget.pt artifacts/model_nocl_noforget.pt` scores the models and the baselines on the test cut, with paired bootstrap intervals, and writes `reports/test_results.md` and `.json`; `--label any_fail` writes the old-label results.
5. `python -m src.export_serving` propagates the trained model on each student's full history and writes `artifacts/serving.npz`, the file the API loads. It checks that the numpy predictor gives the same scores as the PyTorch model.
6. `pytest` runs the tests (split without leakage, metrics, model, serving predictor, API).

The EdNet pipeline of Weeks 2 and 3 still runs with `REVISE_MALIN_DATASET=ednet` (`python -m src.prepare_ednet --kt1 ... --questions ...`, then the same commands); its outputs go to `data/processed/ednet/` and `artifacts/ednet/`. The scripts `check_week2.py`, `sample_users.py`, `build_week3_report.py` and `make_synthetic.py` produced that evidence.

## How it works

- **Data** (`src/prepare_data.py`). One row per (answer, knowledge component): a question tagged with two KCs gives two rows. 718 of the 1,264 students have at least 125 answers (136,130 answers, error rate 23%).
- **Split without leakage.** Each student's answers are sorted by time. Three label windows of 25 answers (train, validation, test) sit at the end of the history, with at least 50 answers before the first one. For each window, inputs are everything strictly before it. Cuts follow timestamps, so answers given at the same moment are never split.
- **Labels** (`window_labels` in `src/build_graph.py`). A notion attempted in the window is positive when the student's error rate on it is above their own error rate over the window (`above_mean`, default), else negative. `REVISE_MALIN_LABEL=any_fail` gives the Week 2-3 label (failed at least once).
- **Model** (`src/model.py`). LightGCN on a student-notion graph (edge = past error), the NLGCL neighbour-layer contrastive loss, and a time term in two places: edges weighted by 2^(-Δt/h) and a score term β(1 - 2^(-Δt/h)), with one learned half-life h per notion and a learned value for notions never practised, plus a learned bias per notion (its difficulty).
- **Serving** (`src/export_serving.py`, `src/predict.py`). At export time the learned embeddings are propagated once on the graph of each student's full history. The API keeps them fixed and recomputes the time term at every request. It uses numpy only.
- **Evaluation** (`src/evaluate.py`, `src/evaluate_test.py`). Only notions the student attempted in the window are ranked. Metrics: NDCG@10, Recall@20, Recall@5, and Recall@10 on failed notions not practised for more than 14 days. Baselines: random, most-popular, global error rate, personal error rate, recent activity.

## Project structure

```
src/        config, prepare_data (DBE-KT22), prepare_ednet (Weeks 2-3), questions, build_graph, model, train,
            evaluate, evaluate_test, export_serving, predict
api/        FastAPI app and SQLite storage
app/        Streamlit interface (talks to the API only)
artifacts/  serving.npz (served model), training histories and logs (three seeds); any_fail_label/ for the
            first DBE runs with the old label; ednet_week3/ for the Week 3 runs
reports/    test results (both labels), seed summary, figures; ednet_week3/ and the Week 3 report for EdNet
scripts/    DBE-KT22 download, Week 2-3 evidence scripts (EdNet)
sim/        toy simulations behind the design choices
tests/      pytest suite
```

## Data licence and references

DBE-KT22: Abdelrahman, Abdelfattah, Wang and Lin, *Database Exercises for Knowledge Tracing (DBE-KT22)*, 2022, Australian Data Archive, [doi:10.26193/6DZWOH](https://doi.org/10.26193/6DZWOH), with the paper *DBE-KT22: A Knowledge Tracing Dataset Based on Online Student Evaluation* ([arXiv:2208.12651](https://arxiv.org/abs/2208.12651)). We acknowledge the authors and the Australian Data Archive; they bear no responsibility for our analysis or its interpretation.

The data is **not** under an open licence. The archive's terms of use allow analytical, non-commercial use and the publication of outputs (statistics, tables, graphs), and forbid giving other people access to the data, in whole or in part. Get the files from the [archive page](https://doi.org/10.26193/6DZWOH) yourself. The download script uses a third-party mirror on [Hugging Face](https://huggingface.co/datasets/Unggi/dbe-kt22_raw_data) for convenience; `artifacts/serving.npz` still contains per-student counts and the notion names and descriptions, which the terms may not allow us to publish (open point for the team). We use the data for coursework only.

- Xu et al., *NLGCL: Naturally Existing Neighbor Layers Graph Contrastive Learning for Recommendation*, RecSys 2025 ([code](https://github.com/Jinfeng-Xu/NLGCL)).
- He et al., *LightGCN*, SIGIR 2020.
- Settles & Meeder, *A Trainable Spaced Repetition Model for Language Learning*, ACL 2016.
- Choi et al., *EdNet: A Large-Scale Hierarchical Dataset in Education*, 2020 (Weeks 2-3, CC BY-NC 4.0).
