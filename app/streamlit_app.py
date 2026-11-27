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

import html
import os
import random
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

API = os.environ.get("REVISE_MALIN_API", "http://127.0.0.1:8000")
# Feedback buttons: label and Material icon (bundled with Streamlit, so they work offline)
ACTIONS = {
    "revised": ("Revised", ":material/check:"),
    "already_known": ("Already known", ":material/thumb_up:"),
    "not_useful": ("Not useful", ":material/block:"),
}
INK, GRAPHITE, HIGHLIGHT_DARK, SLATE = "#1B2A4A", "#5E6B82", "#E8B500", "#9FB0CC"

st.set_page_config(page_title="Révise Malin", page_icon=":material/ink_highlighter:", layout="wide")
st.markdown(f"<style>{(Path(__file__).parent / 'style.css').read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True)


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


def practice_dialog(student_id: int, topic: int, label: str, k: int, days_ahead: float, practised_only: bool) -> None:
    """Open the practice window of a notion (its title names the notion)."""
    st.dialog(f"Practise {label}", width="large")(practice_window)(student_id, topic, label, k, days_ahead,
                                                                    practised_only)


def practice_window(student_id: int, topic: int, label: str, k: int, days_ahead: float, practised_only: bool) -> None:
    """A practice question in a window over the list: answer, correction, how the list moved, next question."""
    state = st.session_state.setdefault("dialog", {})
    if state.get("topic") != topic or state.get("student") != student_id:
        state.clear()
        state.update(topic=topic, student=student_id)
    if "question" not in state:
        qs = api_get("/questions", student_id=student_id, topic=topic, n=1)
        state["question"] = qs[0] if qs else None
    q = state["question"]
    if q is None:
        st.info(f"No other practice question left for {label}.")
    else:
        st.caption(f"Question {q['question']}, difficulty {q['difficulty']} of 3. Notions: "
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
            lines = []
            for name, old, new in state["moves"]:
                out = (", leaves your list" if old and old <= k and (new is None or new > k)
                       else ", enters your list" if new and new <= k and (old is None or old > k) else "")
                way = "rm-up" if (new or 99) < (old or 99) else "rm-down" if (new or 99) > (old or 99) else ""
                lines.append(f"<b>{html.escape(name)}</b> <span class='{way}'>{rank_word(old)} → "
                             f"{rank_word(new)}{out}</span>")
            st.markdown("<div class='rm-moves'>How your list moved<br>" + "<br>".join(lines) + "</div>",
                        unsafe_allow_html=True)
            st.session_state.toast = "; ".join(f"{n} {rank_word(o)} → {rank_word(w)}"
                                               for n, o, w in state["moves"])
    with st.container(horizontal=True, gap="small"):
        if q is not None and "result" in state and st.button(f"Another question on {label}", type="primary",
                                                             icon=":material/edit:"):
            state.pop("question", None)
            state.pop("result", None)
            st.rerun(scope="fragment")
        if st.button("Back to my list", icon=":material/format_list_numbered:"):
            state.clear()
            st.rerun()


def why(r: dict) -> str:
    """The model's reasons, without the error count the row already shows."""
    parts = r["reason"].rstrip(".").split("; ")
    if r["attempts"] and len(parts) > 1:
        parts = parts[1:]
    text = "; ".join(parts)
    return text[:1].upper() + text[1:] + "."


def score_chart(recs: pd.DataFrame) -> alt.Chart:
    """Stacked bars: the three parts of each notion's score, in the order of the list. Not zoomable."""
    parts = {"graph_score": "Students who fail the same notions", "difficulty_score": "How hard it is for everyone",
             "time_score": "Time since you practised it"}
    long = recs.melt(id_vars=["rank", "label"], value_vars=list(parts), var_name="part", value_name="value")
    long["part"] = long["part"].map(parts)
    long["order"] = long["part"].map({v: i for i, v in enumerate(parts.values())})
    order = recs.sort_values("rank")["label"].tolist()
    return (
        alt.Chart(long)
        .mark_bar(height={"band": 0.62})
        .encode(
            y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=220, labelPadding=8)),
            x=alt.X("sum(value):Q", title="Score (higher = revise first)", axis=alt.Axis(tickCount=6, grid=True)),
            color=alt.Color("part:N", sort=list(parts.values()),
                            scale=alt.Scale(domain=list(parts.values()), range=[INK, SLATE, HIGHLIGHT_DARK]),
                            legend=alt.Legend(title=None, orient="top", direction="horizontal", symbolType="square")),
            order=alt.Order("order:Q"),
            tooltip=[alt.Tooltip("label:N", title="Notion"), alt.Tooltip("part:N", title="Part"),
                     alt.Tooltip("value:Q", title="Value", format=".3f")],
        )
        .properties(height=28 * len(recs) + 30)
        .configure(font="Figtree", background="transparent")
        .configure_view(stroke=None)
        .configure_axis(labelColor=INK, titleColor=GRAPHITE, gridColor="#E4E9F0", domainColor="#C9D2DE",
                        tickColor="#C9D2DE", labelFontSize=13, titleFontSize=12, titleFontWeight="normal")
        .configure_legend(labelColor=INK, labelFontSize=13, symbolSize=120, labelLimit=260)
    )


health = api_get("/health")
n_students = health["students"]
model = health["model"]

# ---------- Sidebar: who and how ----------
st.sidebar.markdown("<div class='rm-brand'><mark>Révise</mark> Malin</div>"
                    "<div class='rm-tagline'>See which notions of the Database Systems course you are most "
                    "likely to get wrong, and practise them first.</div>", unsafe_allow_html=True)
if "student" not in st.session_state:
    st.session_state.student = 0
student = st.sidebar.number_input(f"Student (0 to {n_students - 1})", 0, n_students - 1, key="student")
st.sidebar.button("Pick a random student", icon=":material/casino:", type="tertiary",
                  on_click=lambda: st.session_state.update(student=random.randrange(n_students)))
if st.session_state.get("shown_student") != student:  # new student: close any open question
    for key in ("dialog", "toast"):
        st.session_state.pop(key, None)
    st.session_state.shown_student = student
st.sidebar.divider()
k = st.sidebar.slider("Notions to show", 5, 20, 10)
days_ahead = st.sidebar.slider(
    "Revise in … days", 0, 60, 0,
    help="Moves time forward and recomputes the learned time term. On DBE-KT22 the model learned beta > 0: "
         "a notion gets riskier the longer it is not practised (half-life about 7 days), so waiting raises "
         "the notions you have not practised for a while. Notions never practised keep their learned risk.")
practised_only = st.sidebar.toggle("Only notions I have practised", value=False)
st.sidebar.divider()
st.sidebar.markdown(
    f"<div class='rm-footnote'>Model: {html.escape(str(model.get('model')))}, epoch {model.get('epoch')}, "
    f"validation NDCG@10 {model.get('best_val_ndcg@10') or 0:.3f}.<br>Data: DBE-KT22, Australian National "
    f"University (Australian Data Archive).<br>API: <a href='{API}/docs'>{API}/docs</a></div>",
    unsafe_allow_html=True)

# ---------- Student header ----------
profile = api_get(f"/students/{student}")
st.title(f"Student {student}")
st.markdown(
    f"<div class='rm-profile'><b>{profile['answers']:,}</b> answers, <b>{100 * profile['error_rate']:.0f}%</b> wrong, "
    f"<b>{profile['topics_practised']}</b> of {health['topics']} notions practised. "
    f"DBE-KT22 student {profile['dataset_student']}.</div>", unsafe_allow_html=True)

with st.expander("Your history: wrong answers by question difficulty, and weakest notions"):
    left, right = st.columns(2, gap="large")
    levels = pd.DataFrame(profile["by_level"]).dropna(subset=["error_rate"])
    left.caption("Share of wrong answers by question difficulty")
    level_chart = (
        alt.Chart(levels)
        .mark_bar(color=INK, size=46)
        .encode(x=alt.X("level:N", sort=["Easy", "Medium", "Hard"], title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("error_rate:Q", title=None, axis=alt.Axis(format="%", tickCount=4)),
                tooltip=[alt.Tooltip("level:N", title="Difficulty"),
                         alt.Tooltip("error_rate:Q", title="Wrong", format=".0%"),
                         alt.Tooltip("answers:Q", title="Answers")])
        .properties(height=200)
        .configure(font="Figtree", background="transparent")
        .configure_view(stroke=None)
        .configure_axis(labelColor=INK, gridColor="#E4E9F0", domainColor="#C9D2DE", tickColor="#C9D2DE",
                        labelFontSize=13)
    )
    left.altair_chart(level_chart, width="stretch", theme=None)
    right.caption("Highest error rates so far (at least 3 answers)")
    weakest = pd.DataFrame(profile["weakest_topics"])
    if not weakest.empty:
        weakest = weakest.rename(columns={"label": "Notion", "errors": "Wrong", "attempts": "Answers"})
        right.dataframe(weakest[["Notion", "Wrong", "Answers"]], hide_index=True, width="stretch")

# ---------- Ranked notions ----------
res = api_get("/recommend", student_id=student, k=k, days_ahead=days_ahead, practised_only=practised_only)
recs = pd.DataFrame(res["recommendations"])
st.header("Revise these first")
st.markdown(
    f"<div class='rm-lead'>Ranked by the trained model in {res['latency_ms']} ms. Each score adds three parts: "
    "students who fail the same notions as you, how hard the notion is for everyone, and how long since you "
    "practised it. Practise a notion: a wrong answer moves it up, a right one moves it down.</div>",
    unsafe_allow_html=True)

if st.session_state.get("toast"):  # back from a practice window: how the answer moved the list
    st.toast(st.session_state.pop("toast"), icon=":material/ink_highlighter:")

if recs.empty:
    st.info("Every notion has feedback, so the list is empty. Reset your feedback at the bottom of the page.")
else:
    with st.expander("Why this order: the three parts of each score", expanded=True):
        st.altair_chart(score_chart(recs), width="stretch", theme=None)

    for r in res["recommendations"]:
        t = r["topic"]
        strength = max(0.18, min(1.0, (r["priority"] - 60) / 40))  # highlighter strength = priority
        with st.container(key=f"row-{t}"):
            c_rank, c_body, c_act = st.columns([0.9, 6.3, 2.1], gap="medium")
            c_rank.markdown(f"<div class='rm-rank'>{r['rank']}</div>", unsafe_allow_html=True)
            errors = (f"{r['errors']} of {r['attempts']} answers wrong ({100 * r['errors'] / r['attempts']:.0f}%)"
                      if r["attempts"] else "no answer yet")
            c_body.markdown(
                f"<div class='rm-name'><span style='--a:{strength:.2f}'>{html.escape(r['label'])}</span></div>"
                f"<div class='rm-meta'><span>{errors}</span>"
                f"<span class='rm-quiet'>priority {r['priority']:.0f} of 100</span></div>"
                f"<div class='rm-why'>{html.escape(why(r))}</div>", unsafe_allow_html=True)
            with c_body.container(horizontal=True, gap="small"):
                with st.popover("What is it?", type="tertiary", icon=":material/info:"):
                    st.write(r["description"] or "DBE-KT22 gives no description for this notion.")
                    if r["related"]:
                        st.caption("Related notions: " + ", ".join(r["related"]))
                for action, (label, icon) in ACTIONS.items():
                    if st.button(label, key=f"{action}-{t}", icon=icon, type="tertiary"):
                        api_send("POST", "/log_interaction",
                                 json={"student_id": int(student), "topic": t, "action": action})
            if health.get("practice") and r.get("practice_questions", 0) > 0:
                if c_act.button("Practise", key=f"practise-{t}", type="primary", icon=":material/edit:",
                                width="stretch"):
                    practice_dialog(int(student), t, r["label"], k, days_ahead, practised_only)
            elif health.get("practice"):
                c_act.markdown("<div class='rm-none'>No question to practise here: the remaining ones need the "
                               "course worksheet tables, which DBE-KT22 does not include.</div>",
                               unsafe_allow_html=True)

# ---------- Feedback stored in the database ----------
st.header("Your feedback")
logged = api_get(f"/interactions/{student}")
if logged:
    st.caption("Notions you gave feedback on leave your list. The clicks are stored in data/feedback.db "
               "for the next training.")
    table = pd.DataFrame(logged)
    local = datetime.now().astimezone().tzinfo  # SQLite stores UTC; show the time of this computer
    table = table.assign(When=pd.to_datetime(table["created_at"], utc=True).dt.tz_convert(local).dt.strftime("%H:%M"),
                         Notion=table.get("label", table["topic"]),
                         Feedback=table["action"].map(lambda a: ACTIONS.get(a, (a, ""))[0]))
    st.dataframe(table[["When", "Notion", "Feedback"]], hide_index=True, width="stretch")
    if st.button("Reset my feedback and answers", icon=":material/restart_alt:"):
        requests.delete(f"{API}/answers/{student}", timeout=10)
        api_send("DELETE", f"/interactions/{student}")
else:
    st.caption("No feedback yet. A notion you mark as revised, already known or not useful leaves your list; "
               "the clicks are stored in data/feedback.db for the next training.")
    if res.get("session_answers") and st.button(f"Reset my {res['session_answers']} practice answer(s)",
                                                icon=":material/restart_alt:"):
        api_send("DELETE", f"/answers/{student}")
