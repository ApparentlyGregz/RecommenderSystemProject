"""Révise Malin interface (Streamlit). Every number on screen comes from the FastAPI backend.

Run with `python run.py` (starts the API and this interface), or on its own:
    streamlit run app/streamlit_app.py      (the API must be running on REVISE_MALIN_API)
"""
from __future__ import annotations

import os
import time

import requests
import streamlit as st

API = os.environ.get("REVISE_MALIN_API", "http://127.0.0.1:8000").rstrip("/")
TIME_OPTIONS = {"15 min": 3, "30 min": 5, "1 hour": 8}

st.set_page_config(page_title="Révise Malin", page_icon=":books:", layout="wide")


# ------------------------------------------------------------------ API helpers
def api(method: str, path: str, **kw):
    try:
        r = requests.request(method, f"{API}{path}", timeout=10, **kw)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as err:
        st.error(f"The API at {API} did not answer ({err}). Start everything with `python run.py`.")
        st.stop()


@st.cache_data(ttl=300)
def load_students() -> list[dict]:
    return api("GET", "/students", params={"limit": 200})


@st.cache_data(ttl=300)
def load_health() -> dict:
    return api("GET", "/health")


def render_question(q: dict) -> None:
    """Question text: HTML blocks as they are, relational algebra as LaTeX."""
    for block in q["blocks"]:
        if block["type"] == "latex":
            st.latex(block["content"])
        else:
            st.markdown(block["content"], unsafe_allow_html=True)


# ------------------------------------------------------------------ sidebar
health = load_health()
students = load_students()
by_id = {s["student_id"]: s for s in students}
with st.sidebar:
    st.title("Révise Malin")
    st.caption("Revise first what you are most likely to get wrong.")
    student_id = st.selectbox(
        "Student", list(by_id),
        format_func=lambda s: f"Student {s}  ·  {by_id[s]['answers']} answers  ·  {by_id[s]['error_rate']:.0%} wrong",
    )
    minutes = st.radio("Time available today", list(TIME_OPTIONS), index=1, horizontal=True)
    st.divider()
    if st.button("Replay from the dataset history", help="Forget this student's answers and clicks in the app"):
        api("POST", f"/reset/{student_id}")
        st.session_state.clear()
        st.rerun()
    st.caption(f"Model: {health['model_name']} · {health['students']} students · {health['skills']} skills · "
               "data: DBE-KT22, database course at ANU")

if st.session_state.get("student") != student_id:  # new student: close any open exercise
    st.session_state.clear()
    st.session_state["student"] = student_id

# ------------------------------------------------------------------ plan
t0 = time.perf_counter()
plan = api("GET", "/recommend", params={"student_id": student_id, "k": TIME_OPTIONS[minutes]})
elapsed_ms = (time.perf_counter() - t0) * 1000

st.header("Your revision plan")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Answers in your history", by_id[student_id]["answers"])
c2.metric("Error rate", f"{by_id[student_id]['error_rate']:.0%}")
c3.metric("Answered here today", plan["session_answers"])
c4.metric("Response time", f"{elapsed_ms:.0f} ms", help=f"Model time on the server: {plan['latency_ms']} ms")
st.caption(f"The {len(plan['recommendations'])} skills you are most at risk on, for {minutes} of revision. "
           "Priority = rank of the model's risk among all skills (1 = riskiest); your answers today update it.")

last = st.session_state.get("last")
if last:  # correction of the last answer, shown here because the ranking may have moved the skill
    with st.container(border=True):
        res, q = last["result"], last["question"]
        st.markdown(f"**Your answer to question {q['question']}** ({last['skill']}): {last['chosen']}")
        good = [c["text"] for c in q["choices"] if c["id"] in res["correct_choices"]]
        if res["correct"]:
            st.success("Correct. This skill moves down your plan.")
        else:
            st.error(f"Not quite. The correct answer: {' / '.join(good)}. This skill moves up your plan.")
        if res["explanation"]:
            st.info(res["explanation"])
        n1, n2 = st.columns(2)
        if n1.button(f"Another question on {last['skill']}", type="primary"):
            st.session_state["open"] = last["topic"]
            st.session_state.pop("last", None)
            st.rerun()
        if n2.button("Back to my plan"):
            st.session_state.pop("last", None)
            st.rerun()

for rank, r in enumerate(plan["recommendations"], 1):
    topic = r["topic"]
    with st.container(border=True):
        head, bar = st.columns([3, 1])
        head.subheader(f"{rank}. {r['skill']}")
        bar.progress(min(max(r["risk"], 0.0), 1.0), text=f"priority {r['risk']:.2f}")
        if r["description"]:
            st.caption(r["description"])
        st.markdown("\n".join(f"- {reason}" for reason in r["reasons"]))

        b1, b2, b3, b4 = st.columns(4)
        if b1.button("Practise now", key=f"practise-{topic}", type="primary"):
            st.session_state["open"] = topic
            st.session_state.pop("last", None)
            st.session_state.pop("question", None)
        if b2.button("I revised it", key=f"rev-{topic}"):
            api("POST", "/log_interaction", json={"student_id": student_id, "topic": topic, "action": "revised"})
            st.rerun()
        if b3.button("I already know it", key=f"known-{topic}"):
            api("POST", "/log_interaction", json={"student_id": student_id, "topic": topic, "action": "already_known"})
            st.rerun()
        if b4.button("Not useful", key=f"nu-{topic}"):
            api("POST", "/log_interaction", json={"student_id": student_id, "topic": topic, "action": "not_useful"})
            st.rerun()

        if st.session_state.get("open") == topic:
            if "question" not in st.session_state:
                qs = api("GET", "/questions", params={"student_id": student_id, "topic": topic, "n": 1})
                st.session_state["question"] = qs[0] if qs else None
            q = st.session_state["question"]
            if q is None:
                st.info("You have answered every question of this skill in this session.")
                continue
            st.markdown(f"**Question {q['question']}** · difficulty {q['difficulty']}/3 · skills: "
                        + ", ".join(t["skill"] for t in q["topics"]))
            render_question(q)
            if q.get("needs_context"):
                st.warning("This question uses tables from the course worksheet, which the DBE-KT22 dataset "
                           "does not include. It is shown because every other question of this skill was used.")
            text_of = {c["id"]: c["text"] for c in q["choices"]}
            choice = st.radio("Your answer", list(text_of), index=None, key=f"choice-{q['question']}",
                              format_func=text_of.get)
            cols = st.columns([1, 3])
            if cols[0].button("Check my answer", key=f"check-{q['question']}", disabled=choice is None):
                result = api("POST", "/answer", json={"student_id": student_id, "question": q["question"], "choice": choice})
                st.session_state["last"] = {"result": result, "question": q, "topic": topic, "skill": r["skill"],
                                            "chosen": text_of[choice]}
                st.session_state.pop("open", None)
                st.session_state.pop("question", None)
                st.rerun()
            if q["hint"]:
                with cols[1].expander("Hint"):
                    st.write(q["hint"])

if plan["hidden_skills"]:
    st.caption(f"{len(plan['hidden_skills'])} skill(s) hidden because you marked them as known or not useful.")
