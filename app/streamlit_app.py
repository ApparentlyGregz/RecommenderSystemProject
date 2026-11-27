"""Streamlit interface of Révise Malin.

The interface only talks to the FastAPI backend (no model code here):
    GET  /health              loaded model
    GET  /students/{id}       history summary
    GET  /recommend           ranked knowledge components from the trained model
    POST /log_interaction     feedback buttons, stored in SQLite
    GET  /interactions/{id}   stored feedback, DELETE to reset it

Run (the API must be running):
    streamlit run app/streamlit_app.py
or simply `python run.py`, which starts both.
"""
from __future__ import annotations

import os
import random

import pandas as pd
import requests
import streamlit as st

API = os.environ.get("REVISE_MALIN_API", "http://127.0.0.1:8000")
ACTIONS = {"revised": "✅ Revised", "already_known": "👍 Already known", "not_useful": "🚫 Not useful"}

st.set_page_config(page_title="Révise Malin", page_icon="📚", layout="wide")


def api_get(path: str, **params) -> dict | list:
    """GET on the API; stops the page with a clear message if the API is down."""
    try:
        r = requests.get(f"{API}{path}", params=params, timeout=10)
    except requests.ConnectionError:
        st.error(f"The API is not reachable at {API}. Start it with `python run.py`.")
        st.stop()
    if r.status_code != 200:
        st.error(f"API error {r.status_code}: {r.json().get('detail', r.text)}")
        st.stop()
    return r.json()


def api_send(method: str, path: str, **kwargs) -> None:
    """POST/DELETE on the API, then refresh the page so the ranking is recomputed."""
    r = requests.request(method, f"{API}{path}", timeout=10, **kwargs)
    if r.status_code != 200:
        st.error(f"API error {r.status_code}: {r.text}")
    else:
        st.rerun()


health = api_get("/health")
n_students = health["students"]
model = health["model"]

# ---------- Sidebar: who and how ----------
st.sidebar.title("📚 Révise Malin")
st.sidebar.caption("Database Systems course: revise first the notions you are most likely to fail.")
if "student" not in st.session_state:
    st.session_state.student = 0
if st.sidebar.button("🎲 Random student"):
    st.session_state.student = random.randrange(n_students)
student = st.sidebar.number_input(f"Student (0 to {n_students - 1})", 0, n_students - 1, key="student")
k = st.sidebar.slider("Notions to show", 5, 20, 10)
practised_only = st.sidebar.checkbox("Only notions I have already practised", value=False)
days_ahead = st.sidebar.slider(
    "Revise in … days", 0, 60, 0,
    help="Moves time forward and recomputes the learned time term. On DBE-KT22 the model learned that the "
         "notions practised right now are the riskiest (beta < 0), so waiting lowers their priority.")
st.sidebar.divider()
st.sidebar.caption(f"Model **{model.get('model')}** · epoch {model.get('epoch')} · "
                   f"validation NDCG@10 {model.get('best_val_ndcg@10') or 0:.3f} · API {API}")

# ---------- Student profile ----------
profile = api_get(f"/students/{student}")
st.title(f"Student {student}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("DBE-KT22 id", profile["dataset_student"])
c2.metric("Answers", f"{profile['answers']:,}")
c3.metric("Notions practised", f"{profile['topics_practised']} / {health['topics']}")
c4.metric("Error rate", f"{100 * profile['error_rate']:.1f}%")

with st.expander("History: errors per question difficulty and weakest notions", expanded=False):
    left, right = st.columns(2)
    levels = pd.DataFrame(profile["by_level"]).dropna(subset=["error_rate"])
    left.caption("Error rate per question difficulty")
    left.bar_chart(levels.set_index("level")["error_rate"], sort=False)
    right.caption("Highest error rates so far (at least 3 answers)")
    right.dataframe(pd.DataFrame(profile["weakest_topics"]), hide_index=True)

# ---------- Recommendations ----------
res = api_get("/recommend", student_id=student, k=k, days_ahead=days_ahead, practised_only=practised_only)
recs = pd.DataFrame(res["recommendations"])
st.subheader("Notions to revise first")
st.caption(f"Ranked by the trained model ({res['model']}: NLGCL graph + learned time term), "
           f"answered in {res['latency_ms']} ms. Score = graph part (students who fail the same notions) "
           "+ time part (when you last practised it).")

if recs.empty:
    st.info("No notion left: every notion got feedback. Reset the feedback below.")
else:
    chart = recs.set_index("label")[["graph_score", "time_score"]].rename(
        columns={"graph_score": "Graph (NLGCL)", "time_score": "Time term"})
    st.bar_chart(chart, horizontal=True, stack=True, sort=False)

    for r in res["recommendations"]:
        with st.container(border=True):
            left, right = st.columns([4, 3])
            dsp = "not yet" if r["days_since_practice"] is None else f"{r['days_since_practice']} days ago"
            text = (f"**#{r['rank']} · {r['label']}**  \n"
                    f"Priority {r['priority']:.0f}/100 · errors {r['errors']} / answers {r['attempts']} · "
                    f"last practised {dsp}  \n_{r['reason']}_")
            left.markdown(text)
            with left.popover("What is it?"):
                st.write(r["description"] or "No description in DBE-KT22.")
                if r["related"]:
                    st.caption("Related notions: " + ", ".join(r["related"]))
            cols = right.columns(len(ACTIONS))
            for col, (action, label) in zip(cols, ACTIONS.items()):
                if col.button(label, key=f"{action}-{r['topic']}"):
                    api_send("POST", "/log_interaction",
                             json={"student_id": int(student), "topic": r["topic"], "action": action})

# ---------- Feedback stored in the database ----------
st.subheader("Your feedback (stored in SQLite)")
logged = api_get(f"/interactions/{student}")
if logged:
    st.dataframe(pd.DataFrame(logged)[["created_at", "topic", "action"]], hide_index=True)
    if st.button("Reset my feedback"):
        api_send("DELETE", f"/interactions/{student}")
else:
    st.caption("No feedback yet. A notion you give feedback on leaves your list; the clicks are kept "
               "in data/feedback.db as new interactions for the next training.")
