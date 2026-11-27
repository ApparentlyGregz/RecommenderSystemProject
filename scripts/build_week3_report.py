"""Build the Week 3 status report (.docx) with the layout of the Week 1 submission.

The Week 1 file is used as a template: its styles (Aptos, Title, Heading 1/2,
List Bullet, Table Grid), page setup and footer are kept; the body is replaced.
Test metrics are read from reports/test_results.json (src.evaluate_test), so the
report is rebuilt with the final numbers by running this script again.

Usage:
    python scripts/build_week3_report.py --commit <short hash>
    ->  reports/Revise_Malin_Week3_Status_Report.docx
"""
from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "Revise_Malin_Project_Plan_Submission.docx"
RESULTS = ROOT / "reports" / "test_results.json"
FIGURE = ROOT / "reports" / "figures" / "training_curves_week3_compact.png"
OUT = ROOT / "reports" / "Revise_Malin_Week3_Status_Report.docx"
DATE = "22 November 2026"
HEADER_FILL = "E8EEF5"  # header shading of the Week 1 tables
MUTED = RGBColor(0x5F, 0x66, 0x6E)

ROWS = [  # (key in test_results.json, label in the report, is a trained model)
    ("random", "Random (mean of 10 seeds)", False),
    ("most_popular", "Most-popular (attempt frequency)", False),
    ("global_error", "Global error rate", False),
    ("personal_error", "Personal error rate", False),
    ("recent_activity", "Recent activity (answers in the last 100)", False),
    ("nocl_noforget", "LightGCN", True),
    ("nocl_forget", "LightGCN + forgetting", True),
    ("cl_noforget", "NLGCL, no forgetting", True),
    ("full", "Full model: NLGCL + forgetting", True),
]


# ---------------------------------------------------------------- helpers
def rich(par, text: str, size: float | None = None, color: RGBColor | None = None, italic: bool = False):
    """Add text to a paragraph; **double stars** mark bold spans."""
    for i, chunk in enumerate(re.split(r"\*\*", text)):
        if not chunk:
            continue
        run = par.add_run(chunk)
        run.bold = i % 2 == 1 or None
        run.italic = italic or None
        if size:
            run.font.size = Pt(size)
        if color is not None:
            run.font.color.rgb = color
    return par


def para(doc, text: str, style: str | None = None, **kw):
    return rich(doc.add_paragraph(style=style), text, **kw)


def bullet(doc, text: str):
    p = rich(doc.add_paragraph(style="List Bullet"), text)
    pf = p.paragraph_format  # same indentation and spacing as the Week 1 bullets
    pf.space_after = Pt(3)
    pf.left_indent, pf.first_line_indent = Pt(12.95), Pt(-7.9)
    return p


def shade(cell, fill: str) -> None:
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(shd)


def table(doc, header: list[str], rows: list[list[str]], widths: list[float], bold_rows: set[int] = frozenset()):
    """Week 1 table style: Table Grid, centred, 8 pt, shaded bold header, tight cell spacing."""
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for r, values in enumerate([header] + rows):
        for c, value in enumerate(values):
            cell = t.cell(r, c)
            cell.width = Inches(widths[c])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.0
            if c > 0 and header[c] in {"NDCG@10", "Recall@20", "Recall@10 old topics"}:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            rich(p, f"**{value}**" if (r == 0 or (r - 1) in bold_rows) and value else value, size=8)
            if r == 0:
                shade(cell, HEADER_FILL)
    for c, w in enumerate(widths):  # fixed grid, as in the Week 1 tables
        t.columns[c].width = Inches(w)
    for row in t.rows[:-1]:  # keep the whole table on one page
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.keep_with_next = True
    return t


def spacer(doc, pt: float = 2):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.line_spacing = Pt(pt)


def fmt(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def gap_text(gaps: list[dict], model: str, ref: str, metric: str) -> str:
    for g in gaps:
        if g["model"] == model and g["reference"] == ref and g["metric"] == metric:
            return f"{g['gap']:+.3f} [{g['ci_low']:+.3f}, {g['ci_high']:+.3f}]"
    return "pending"


# ---------------------------------------------------------------- document
def build(commit: str) -> Path:
    res = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {"metrics": {}, "gaps": [], "forgetting": {}}
    m, gaps, forget, floor = res["metrics"], res["gaps"], res.get("forgetting", {}), res.get("sanity_floor", {})
    trained = [k for k, _, is_model in ROWS if is_model and k in m]
    best = max(trained, key=lambda k: m[k]["ndcg@10"]) if trained else None
    best_label = dict((k, lab) for k, lab, _ in ROWS).get(best, "best model")
    fp = forget.get("full") or forget.get("nocl_forget") or {}
    n_students = int(m["random"]["students"]) if "random" in m else 4988
    n_old = int(m["random"]["students_old"]) if "random" in m else 3480

    doc = Document(TEMPLATE)
    body = doc.element.body
    sect = copy.deepcopy(body.find(qn("w:sectPr")))
    for child in list(body):
        body.remove(child)
    body.append(sect)
    for p in doc.sections[0].footer.paragraphs:  # keep the Week 1 footer run format, change its text
        if p.runs:
            p.runs[0].text = f"Révise Malin  |  Week 3 status report  |  {DATE}"
            for extra in p.runs[1:]:
                extra.text = ""
    doc.core_properties.title = "Révise Malin Week 3 Status Report"

    para(doc, "Révise Malin Week 3 Status Report", style="Title")
    para(doc, f"Week 3 submission  ·  Recommender Systems  ·  {DATE}  ·  Evidence commit {commit}")
    para(doc, "The model is trained on the full EdNet sample and clears the required baselines. Our checks also show that "
              "most of its accuracy comes from how often topics are shown, and that forgetting does not help on this label.")

    # 1 Progress
    para(doc, "1  Progress update", style="Heading 1")
    table(doc, ["Status", "Item"], [
        ["Done", "EdNet sample processed (5,000 students, 7.07 M answers, 188 topics), reproduced on a second machine."],
        ["Done", "Full model and three ablations trained on the full V1 data, with curves (Figure 1)."],
        ["Done", "Test evaluation with paired bootstrap intervals; added most-popular and recent-activity baselines."],
        ["Done", "Loss-scale bug fixed in our NLGCL code (section 2.2); 19 tests pass, including leakage checks."],
        ["In progress", "predict.py, a real /recommend and the Streamlit screen (a mock-up exists)."],
        ["In progress", "Demo data moved to DBE-KT22 (has question text) and a label corrected for exposure."],
        ["Blocked", "Nothing technical."],
    ], widths=[0.95, 5.95])

    # 2 Results
    para(doc, "2  Key results", style="Heading 1")
    para(doc, "2.1  Training and validation curves (Q1)", style="Heading 2")
    if FIGURE.exists():
        doc.add_picture(str(FIGURE), width=Inches(6.6))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    para(doc, "Figure 1. BPR loss (train, validation) and validation NDCG@10 per epoch; dots mark the kept checkpoint; "
              "dashed gray: original summed InfoNCE. All runs converge: no NaN, no divergence, validation loss flat after "
              "about 20 epochs. Early stopping and the contrastive weight use validation only.",
         size=8, color=MUTED, italic=True)

    para(doc, "2.2  A bug found and fixed in the contrastive loss", style="Heading 2")
    para(doc, "The official NLGCL code sums InfoNCE over the batch. On our graph this made its gradient about **700 times** "
              "the BPR gradient at initialisation: BPR stayed at ln 2 (validation NDCG@10 0.754 after 34 epochs). Masking "
              "false negatives changed nothing. We now average it over the batch, with its weight chosen on validation (1e-3).")

    para(doc, "2.3  Offline metrics on the test cut (Q2)", style="Heading 2")
    para(doc, f"Same split and script for all. Each of the {n_students:,} test students ranks only the topics attempted "
              "in the test window (median 58); paired bootstrap, 95% intervals.")
    rows = [[label, fmt(m[k]["ndcg@10"]), fmt(m[k]["recall@20"]), fmt(m[k]["recall@10_old"])] if k in m
            else [label, "pending", "pending", "pending"] for k, label, _ in ROWS]
    bold = {i for i, (k, _, _) in enumerate(ROWS) if k == best}
    table(doc, ["Method", "NDCG@10", "Recall@20", "Recall@10 old topics"], rows,
          widths=[3.3, 1.15, 1.15, 1.3], bold_rows=bold)
    spacer(doc, 4)
    n_pass, n_models = sum(floor.values()), len(floor)
    para(doc, f"{n_pass} of {n_models} trained models beat random and most-popular on both metrics, with every 95% "
              f"interval above zero. Best model ({'the full model' if best == 'full' else best_label}) against most-popular: "
              f"{gap_text(gaps, best, 'most_popular', 'ndcg@10')} NDCG@10 and "
              f"{gap_text(gaps, best, 'most_popular', 'recall@20')} Recall@20.")
    para(doc, "**The margin is small, and we know why.** Model scores correlate at 0.84 with topic frequency. The label, "
              "\"failed at least once in the next 100 answers\", grows with exposure: 34% of topics seen once in the window "
              "are failed, 95% of those seen more than eight times. A topic the app shows often is likely to be failed by "
              "anyone, so the models learn exposure first and the student second.")

    para(doc, "2.4  Innovation: forgetting component and ablations (Q4)", style="Heading 2")
    beta = fp.get("beta")
    para(doc, f"Own metric: Recall@10 on failed topics not practised for more than 14 days ({n_old:,} test students have "
              f"one). The learned weight β is **negative** ({beta:+.2f}" if beta is not None else
              f"Own metric: Recall@10 on failed topics not practised for more than 14 days ({n_old:,} test students have "
              "one). The learned weight β is **negative** (pending")
    doc.paragraphs[-1].add_run(
        f"; median half-life {fp['half_life_days_median']:.1f} days). " if fp else "). ")
    rich(doc.paragraphs[-1],
         "Recent practice raises the predicted risk instead of old practice: on EdNet, \"practised recently\" means "
         "\"the part the student is drilling now\", where the next errors happen. The forgetting variants therefore rank "
         "old topics lower than the global error rate does (0.301), and below the same models without forgetting. In the "
         f"four-variant ablation the full model is best, but only by {gap_text(gaps, 'full', 'nocl_noforget', 'ndcg@10')} "
         "NDCG@10 over plain LightGCN. Neither component brings a meaningful gain on this label; we report this as the "
         "main finding of the week.")

    para(doc, "2.5  Qualitative check: three test students", style="Heading 2")
    para(doc, "With the full model, three random test students get 9, 9 and 10 failed topics in their top 10 (most-popular: "
              "9, 8 and 9; random: 6.8, 4.7 and 5.9). The first five topics are the same for all three: EdNet tags 181 to "
              "185 (TOEIC Part 2), the tags on the most questions. Personalisation shows in ranks 6 to 10. Over all "
              "students, 146 of 188 topics reach a top 10, yet six reach at least 78% of them: popularity dominates "
              "(reports/sample_users_week3.md).")

    # 3 Live system
    para(doc, "3  Live system (Q5)", style="Heading 1")
    para(doc, "FastAPI runs and /log_interaction writes to SQLite, but /recommend still returns mock scores and the "
              "Streamlit screen is not built, so we cannot show a working end-to-end capture this week.")

    # 4 Risk
    para(doc, "4  Risk re-assessment", style="Heading 1")
    table(doc, ["Question", "Answer", "Evidence"], [
        ["Q1 Training", "Yes", "Figure 1; artifacts/history_*.json; scripts/plot_curves.py"],
        ["Q2 Sanity floor", "Yes", "Table in 2.3; reports/test_results.md with bootstrap intervals"],
        ["Q3 Leakage", "Yes", "Temporal per-student split; graph, error rates and time gaps from answers strictly before "
                              "each cut; validation-only tuning; tests/test_split.py, tests/test_metrics.py"],
        ["Q4 Innovation proof", "Yes (negative)", "Recall@10 on old topics and four-variant ablation, section 2.4"],
        ["Q5 Live system", "No", "Mock /recommend, no interface yet"],
        ["Q6 Reproducibility", "No", "Clean-machine test not done"],
        ["Q7 Schedule", "Behind", "Plan below"],
    ], widths=[1.45, 1.05, 4.4])
    spacer(doc, 4)
    para(doc, "**Risk level: High** (two \"No\" answers and behind schedule). Plan before the showcase:")
    bullet(doc, "**Data:** move the demo to DBE-KT22 (1,264 students, 212 database questions with text and choices, 98 "
                "named skills) so the app shows real questions and readable topics; same pipeline, windows of 30 answers.")
    bullet(doc, "**Label:** replace \"failed at least once\" with a target free of exposure (error rate in the window above "
                "the student's own average), then re-run the ablation with the same baselines.")
    bullet(doc, "**Serving:** predict.py loads exported embeddings with NumPy; real /recommend under 2 s; Streamlit screen "
                "grouped by topic, with reasons and the three feedback buttons.")
    bullet(doc, "**Reproducibility:** a member who did not write the launch code runs the README on a clean machine, "
                "target under 10 minutes.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    return OUT


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commit", default="(to be pushed)")
    print(f"Saved {build(ap.parse_args().commit)}")
