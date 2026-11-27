"""Streamlit interface of Révise Malin.

The interface only talks to the FastAPI backend (no model code here):
    GET  /health              loaded model
    GET  /students/{id}       history summary
    GET  /recommend           ranked knowledge components from the trained model
    POST /log_interaction     feedback buttons, stored in SQLite
    GET  /interactions/{id}   stored feedback, DELETE to reset it
    GET  /questions           a real DBE-KT22 question of a notion (practice mode)
    POST /answer              checks and stores an answer; it moves the notion in the next ranking

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


def api_post(path: str, **kwargs) -> dict:
    """POST on the API and return its JSON (no page refresh)."""
    r = requests.post(f"{API}{path}", timeout=10, **kwargs)
    if r.status_code != 200:
        st.error(f"API error {r.status_code}: {r.text}")
        st.stop()
    return r.json()


def render_question(q: dict) -> None:
    """Question text from DBE-KT22: HTML blocks as they are, relational algebra as LaTeX."""
    for block in q["blocks"]:
        if block["type"] == "latex":
            st.latex(block["content"])
        else:
            st.markdown(block["content"], unsafe_allow_html=True)


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
if st.session_state.get("shown_student") != student:  # new student: close any open question
    for key in ("practice_topic", "practice_q", "last_answer"):
        st.session_state.pop(key, None)
    st.session_state.shown_student = student
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
st.caption(f"Ranked by the trained model ({res['model']}), answered in {res['latency_ms']} ms. "
           "Score = graph part (students who fail the same notions) + difficulty of the notion + time part "
           "(when you last practised it). Practise a notion: each answer moves it, up if wrong, down if right.")

last = st.session_state.get("last_answer")
if last:  # correction of the last practice answer, shown here because the ranking may have moved the notion
    with st.container(border=True):
        a, q = last["result"], last["question"]
        st.markdown(f"**Your answer to question {q['question']}** ({last['label']}): {last['chosen']}")
        good = [c["text"] for c in q["choices"] if c["id"] in a["correct_choices"]]
        if a["correct"]:
            st.success("Correct: this notion moves down your list.")
        else:
            st.error(f"Not quite. The correct answer: {' / '.join(good)}. This notion moves up your list.")
        if a["explanation"]:
            st.info(a["explanation"])
        n1, n2 = st.columns(2)
        if n1.button(f"Another question on {last['label']}", type="primary"):
            st.session_state.practice_topic = last["topic"]
            st.session_state.pop("last_answer")
            st.rerun()
        if n2.button("Close"):
            st.session_state.pop("last_answer")
            st.rerun()

if recs.empty:
    st.info("No notion left: every notion got feedback. Reset the feedback below.")
else:
    chart = recs.set_index("label")[["graph_score", "difficulty_score", "time_score"]].rename(
        columns={"graph_score": "Graph (NLGCL)", "difficulty_score": "Difficulty", "time_score": "Time term"})
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
            if health.get("practice") and right.button("📝 Practise this notion", key=f"practise-{r['topic']}",
                                                       type="primary"):
                st.session_state.practice_topic = r["topic"]
                st.session_state.pop("practice_q", None)
                st.session_state.pop("last_answer", None)
            cols = right.columns(len(ACTIONS))
            for col, (action, label) in zip(cols, ACTIONS.items()):
                if col.button(label, key=f"{action}-{r['topic']}"):
                    api_send("POST", "/log_interaction",
                             json={"student_id": int(student), "topic": r["topic"], "action": action})

            if st.session_state.get("practice_topic") == r["topic"]:
                if "practice_q" not in st.session_state:
                    qs = api_get("/questions", student_id=int(student), topic=r["topic"], n=1)
                    st.session_state.practice_q = qs[0] if qs else None
                q = st.session_state.practice_q
                if q is None:
                    st.info("You answered every question of this notion in this session.")
                    continue
                st.markdown(f"**Question {q['question']}** · difficulty {q['difficulty']}/3 · notions: "
                            + ", ".join(t["label"] for t in q["topics"]))
                render_question(q)
                if q["needs_context"]:
                    st.warning("This question uses tables from the course worksheet, which DBE-KT22 does not "
                               "include. It is shown because no self-contained question of this notion is left.")
                text_of = {c["id"]: c["text"] for c in q["choices"]}
                choice = st.radio("Your answer", list(text_of), index=None, format_func=text_of.get,
                                  key=f"choice-{q['question']}")
                c1, c2 = st.columns([1, 3])
                if c1.button("Check my answer", key=f"check-{q['question']}", disabled=choice is None):
                    result = api_post("/answer", json={"student_id": int(student), "question": q["question"],
                                                       "choice": choice})
                    st.session_state.last_answer = {"result": result, "question": q, "topic": r["topic"],
                                                    "label": r["label"], "chosen": text_of[choice]}
                    for key in ("practice_topic", "practice_q"):
                        st.session_state.pop(key, None)
                    st.rerun()
                if q["hint"]:
                    with c2.expander("Hint"):
                        st.write(q["hint"])

# ---------- Feedback stored in the database ----------
st.subheader("Your feedback (stored in SQLite)")
logged = api_get(f"/interactions/{student}")
if logged:
    st.dataframe(pd.DataFrame(logged)[["created_at", "topic", "action"]], hide_index=True)
    if st.button("Reset my feedback and answers"):
        requests.delete(f"{API}/answers/{student}", timeout=10)
        api_send("DELETE", f"/interactions/{student}")
else:
    st.caption("No feedback yet. A notion you give feedback on leaves your list; the clicks are kept "
               "in data/feedback.db as new interactions for the next training.")
    if res.get("session_answers") and st.button(f"Reset my {res['session_answers']} practice answer(s)"):
        api_send("DELETE", f"/answers/{student}")
