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


def full_ranks(student_id: int, days_ahead: float, practised_only: bool) -> dict[int, int]:
    """Rank of every notion in the student's current list (top 50), to show how an answer moved them."""
    res = api_get("/recommend", student_id=student_id, k=50, days_ahead=days_ahead, practised_only=practised_only)
    return {r["topic"]: r["rank"] for r in res["recommendations"]}


def rank_word(rank: int | None) -> str:
    return f"#{rank}" if rank else "beyond #50"


@st.dialog("Practice", width="large")
def practice_dialog(student_id: int, topic: int, label: str, k: int, days_ahead: float, practised_only: bool) -> None:
    """A practice question in a window over the list: answer, correction, how the list moved, next question."""
    state = st.session_state.setdefault("dialog", {})
    if state.get("topic") != topic or state.get("student") != student_id:
        state.clear()
        state.update(topic=topic, student=student_id)
    if "question" not in state:
        qs = api_get("/questions", student_id=student_id, topic=topic, n=1)
        state["question"] = qs[0] if qs else None
    st.subheader(label)
    q = state["question"]
    if q is None:
        st.info(f"No other practice question left for {label}.")
    else:
        st.caption(f"Question {q['question']} · difficulty {q['difficulty']}/3 · notions: "
                   + ", ".join(t["label"] for t in q["topics"]))
        render_question(q)
        text_of = {c["id"]: c["text"] for c in q["choices"]}
        done = "result" in state
        choice = st.radio("Your answer", list(text_of), index=None, format_func=text_of.get,
                          key=f"dlg-choice-{q['question']}", disabled=done)
        if not done:
            c1, c2 = st.columns([1, 2])
            if c1.button("Check my answer", type="primary", disabled=choice is None):
                before = full_ranks(student_id, days_ahead, practised_only)
                state["result"] = api_post("/answer", json={"student_id": student_id, "question": q["question"],
                                                            "choice": choice})
                after = full_ranks(student_id, days_ahead, practised_only)
                state["moves"] = [(t["label"], before.get(t["topic"]), after.get(t["topic"])) for t in q["topics"]]
                st.rerun(scope="fragment")
            if q["hint"]:
                with c2.expander("Hint"):
                    st.write(q["hint"])
        else:
            a = state["result"]
            good = [c["text"] for c in q["choices"] if c["id"] in a["correct_choices"]]
            if a["correct"]:
                st.success("Correct.")
            else:
                st.error(f"Not quite. The correct answer: {' / '.join(good)}.")
            if a["explanation"]:
                st.info(a["explanation"])
            moves = []
            for name, old, new in state["moves"]:
                out = (" (leaves your list)" if old and old <= k and (new is None or new > k)
                       else " (enters your list)" if new and new <= k and (old is None or old > k) else "")
                moves.append(f"**{name}**: {rank_word(old)} → {rank_word(new)}{out}")
            st.markdown("Your list moved:  \n" + "  \n".join(moves))
            st.session_state.toast = " · ".join(f"{n}: {rank_word(o)} → {rank_word(w)}"
                                                for n, o, w in state["moves"])
    left, right = st.columns(2)
    if q is not None and "result" in state and left.button(f"Another question on {label}", type="primary"):
        state.pop("question", None)
        state.pop("result", None)
        st.rerun(scope="fragment")
    if right.button("Back to my list"):
        state.clear()
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
if st.session_state.get("shown_student") != student:  # new student: close any open question
    for key in ("dialog", "toast"):
        st.session_state.pop(key, None)
    st.session_state.shown_student = student
k = st.sidebar.slider("Notions to show", 5, 20, 10)
practised_only = st.sidebar.checkbox("Only notions I have already practised", value=False)
days_ahead = st.sidebar.slider(
    "Revise in … days", 0, 60, 0,
    help="Moves time forward and recomputes the learned time term. On DBE-KT22 the model learned beta > 0: "
         "a notion gets riskier the longer it is not practised (half-life about 7 days), so waiting raises "
         "the notions you have not practised for a while. Notions never practised keep their learned risk.")
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
           "(when you last practised it). Practise a notion: the question opens in a window, and each answer "
           "moves the notion, up if wrong, down if right.")

if st.session_state.get("toast"):  # back from a practice window: how the answer moved the list
    st.toast(st.session_state.pop("toast"), icon="📝")

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
            if health.get("practice") and r.get("practice_questions", 0) > 0:
                if right.button("📝 Practise this notion", key=f"practise-{r['topic']}", type="primary"):
                    practice_dialog(int(student), r["topic"], r["label"], k, days_ahead, practised_only)
            elif health.get("practice"):
                right.caption("No practice question available for this notion: its DBE-KT22 questions need the "
                              "course worksheet tables, which the dataset does not include (or you answered them all).")
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
    if st.button("Reset my feedback and answers"):
        requests.delete(f"{API}/answers/{student}", timeout=10)
        api_send("DELETE", f"/interactions/{student}")
else:
    st.caption("No feedback yet. A notion you give feedback on leaves your list; the clicks are kept "
               "in data/feedback.db as new interactions for the next training.")
    if res.get("session_answers") and st.button(f"Reset my {res['session_answers']} practice answer(s)"):
        api_send("DELETE", f"/answers/{student}")
