# Révise Malin

Révise Malin ranks the exam topics a student is most likely to fail, so they revise those first. It combines a graph recommender (NLGCL, RecSys 2025) with a learned forgetting curve, trained on TOEIC practice data from EdNet.

Team: Grégoire Champourlier, Maxence Simonet, Hippolyte Nussbaumer, Dan Waintraub (ECE Paris, ING5 Data & AI).

**Status (Week 2):** data pipeline, model forward pass, evaluation harness and a mock API are in place. Training, real predictions and the Streamlit interface come in Week 3.

## Run the API (Python 3.11 to 3.14)

```bash
git clone <repo> && cd <repo>
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python run.py                    # then open http://localhost:8000/docs
```

`/recommend` returns mock scores this week (`"mode": "mock"`). `/log_interaction` already writes to SQLite (`data/feedback.db`, created at launch).

## Rebuild the data and the model

```bash
pip install -r requirements-train.txt
```

1. Download EdNet-KT1 ([bit.ly/ednet_kt1](http://bit.ly/ednet_kt1), 1.2 GB) and Contents ([bit.ly/ednet-content](http://bit.ly/ednet-content)) into `data/raw/`. Do not unzip KT1: the script reads the zip directly.
2. `python -m src.prepare_data --kt1 data/raw/EdNet-KT1.zip --questions data/raw/contents/questions.csv`
   Samples 5,000 students with at least 400 answers and writes `data/processed/` (events, topics, stats).
3. `python -m src.evaluate` checks the harness on mock predictions and prints the baselines on the test cut.
4. `python -m src.train` trains the full model. Ablations: `--no-forgetting`, `--no-contrastive`, or both.
5. `python scripts/check_week2.py` prints the evidence for the Week 2 report into `reports/week2_evidence.txt`.
6. `pytest` runs the tests (split without leakage, metrics, model, API).

No EdNet yet? `python scripts/make_synthetic.py` writes a small fake dataset in the exact EdNet format to `data/synthetic/`, which the same commands accept.

## How it works

- **Split without leakage.** Each student's answers are sorted by time. Three label windows of 100 answers (train, validation, test) sit at the end of the history. For each window, inputs are everything strictly before it. Cuts follow timestamps, so a bundle of questions answered together is never split.
- **Labels.** A topic attempted and failed at least once in the window is positive; attempted and always correct is negative.
- **Model** (`src/model.py`). LightGCN on a student-topic graph (edge = past error), the NLGCL neighbour-layer contrastive loss, and forgetting in two places: edges weighted by 2^(-Δt/h) and a score term β(1 - 2^(-Δt/h)), with one learned half-life h per topic.
- **Evaluation** (`src/evaluate.py`). Only topics the student attempted in the window are ranked (the Santa app chose which questions to show). Metrics: NDCG@10, Recall@20, Recall@5, and Recall@10 on topics not practised for more than 14 days. Baselines: random, global error rate (most-popular), personal error rate.

## Project structure

```
src/        config, prepare_data, build_graph, model, train, evaluate
api/        FastAPI app and SQLite storage
scripts/    synthetic data generator, Week 2 evidence
sim/        toy simulations behind the design choices
tests/      pytest suite
```

## Data licence and references

EdNet is © Riiid, released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) ([github.com/riiid/ednet](https://github.com/riiid/ednet); Choi et al., *EdNet: A Large-Scale Hierarchical Dataset in Education*, 2020). We use it for non-commercial coursework only.

- Xu et al., *NLGCL: Naturally Existing Neighbor Layers Graph Contrastive Learning for Recommendation*, RecSys 2025 ([code](https://github.com/Jinfeng-Xu/NLGCL)).
- He et al., *LightGCN*, SIGIR 2020.
- Settles & Meeder, *A Trainable Spaced Repetition Model for Language Learning*, ACL 2016.
