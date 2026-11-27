# Révise Malin — final presentation (Startup Showcase)

Slide content for Canva. Each slide gives the **title**, the **text on the slide**, the **visual**, and **speaker notes** (what to say, not to put on the slide). About 13 slides for a 5–7 minute pitch, then the live demo.

All numbers come from the repository after PR #3: `README.md`, `reports/test_results.md`, `reports/test_results_any_fail.md`, `reports/seed_summary.md`, `reports/Revise_Malin_Week3_Status_Report.pdf`.

**Suggested look:** dark navy `#14213D` and off-white `#FBFBF8`, one accent (amber `#F2A541` or blue `#2A78D6`). Headings in a bold sans (Space Grotesk or Rubik), text in Inter. One idea per slide, big numbers rather than bullet lists.

---

## Slide 1 — Cover

**Title:** Révise Malin
**Subtitle:** Revise first what you are most likely to get wrong.
**Small text:** Grégoire Champourlier · Maxence Simonet · Hippolyte Nussbaumer · Dan Waintraub — ECE Paris, ING5 Data & AI

**Visual:** app logo or a clean screenshot of the revision list, cropped.

**Notes:** One sentence pitch: “The night before an exam, you don’t have time to revise everything. Révise Malin tells you which notions you are about to fail, lets you practise them on real exam questions, and adapts after every answer.”

---

## Slide 2 — The problem

**Title:** The night before the exam
**On slide (big number + one line):**
- **93 notions** in a database course
- **3 hours** left
- Students revise what they like, not what they will fail

**Visual:** a long checklist of notions with a clock, or a stressed student icon.

**Notes:** Students pick revision topics by feeling. The notions they feel confident about are often the ones they practise, while the risky ones stay untouched. A teacher can’t personalise this for hundreds of students.

---

## Slide 3 — Our solution

**Title:** A revision list that knows you
**On slide (three steps, icons):**
1. **Rank** — the notions you are most likely to fail, with the reason for each
2. **Practise** — real exam questions on that notion, corrected instantly
3. **Adapt** — every answer moves the notion up or down

**Visual:** screenshot of the app (list of notions with “📝 Practise this notion” and an open question).

**Notes:** Each notion says why it is there: “5 errors in 13 answers; students who fail the same notions as you often fail this one too; one of the hardest notions for every student”. A wrong answer pushes the notion up, a right one pushes it down.

---

## Slide 4 — How it works

**Title:** From answers to a ranking in 1 ms
**On slide (pipeline diagram, left to right):**
Student answers → student–notion graph → graph recommender (NLGCL + time term + notion difficulty) → FastAPI → Streamlit app → answers stored in SQLite → back into the ranking

**Small text:** model file 0.4 MB · ranking in about 1 ms on a laptop CPU · no GPU, no PyTorch to run the app

**Visual:** simple flow diagram with 5 boxes and a loop arrow.

**Notes:** The graph links each student to the notions they got wrong. The model learns an embedding per student and per notion from that graph (LightGCN with the NLGCL contrastive loss from RecSys 2025), a difficulty per notion, and a time term with one learned half-life per notion. We export it once; the API ranks with NumPy. Everything the student does in the app is logged and changes the next ranking.

---

## Slide 5 — The data

**Title:** Real students, real questions
**On slide (big numbers):**
- **718** students
- **212** questions with text and answer choices
- **93** named notions (Joins, Foreign keys, Transactions, SQL injection…)
- **136,000** answers, 2019–2022

**Small text:** DBE-KT22, database course at the Australian National University

**Visual:** four big numbers in cards.

**Notes:** We started on EdNet, a TOEIC dataset with 7 million answers, but it has no question text and its topics are anonymous numbers like “Topic 52”. A student can’t do anything with “Topic 52”. We switched to DBE-KT22, which gives the full questions and named notions, so the app can show real exercises.

---

## Slide 6 — Evaluating without cheating

**Title:** We only predict the future
**On slide:**
- Each student’s answers in time order
- Train, validation, test = three windows of 25 answers at the **end** of the history
- The model only sees answers **before** each window
- A notion is “at risk” if the student does **worse than usual** on it

**Visual:** a timeline bar split into history | train | val | test.

**Notes:** No test answer is ever used for training, graph building or tuning; tests in the code check it. The label matters: we showed in Week 3 that “failed at least once” mostly rewards notions that are shown often. With “worse than your own average” the label no longer depends on how often a notion comes up.

---

## Slide 7 — Results

**Title:** Better than every baseline on the ranking
**On slide (table, test set, 710 students):**

| Method | NDCG@10 | Recall@5 |
|---|---|---|
| Random | 0.528 | 0.315 |
| Most-popular | 0.580 | 0.360 |
| Notion difficulty (global error rate) | 0.668 | **0.447** |
| **Révise Malin** | **0.686** | 0.422 |

**Small text:** mean of 3 training seeds (± 0.001); 95% bootstrap intervals in `reports/test_results.md`

**Visual:** horizontal bar chart of NDCG@10 with Révise Malin highlighted.

**Notes:** We beat every baseline on NDCG@10, including the hardest one, the difficulty of each notion: +0.019 with a 95% interval of [+0.008, +0.030], and most-popular by +0.107. The difficulty baseline stays ahead on Recall@5: when you only look at the top 5, the hardest notions overall are a strong bet. Recall@20 is about 0.99 for everyone because a student meets only about 17 notions per window, so we don’t use it.

---

## Slide 8 — What we learned

**Title:** Three honest findings
**On slide (three cards):**
1. **A bug worth 700×** — the contrastive loss from the paper drowned the ranking loss; we found it and fixed it
2. **The label decides the science** — with the right label, forgetting is learned the right way round (half-life ≈ 7 days)
3. **Simple pieces matter most** — a learned difficulty per notion helps; the paper’s contrastive loss adds nothing measurable here

**Visual:** three cards with a small icon each (bug, target, scale).

**Notes:** With “failed at least once”, the model learned that recently practised notions are the riskiest (β = −0.025): that label confuses “currently studying” with “at risk”. With “worse than usual” it learns real forgetting: β = +0.049, a notion gets riskier the longer it is not practised, median half-life 6.9 days. The time term adds +0.010 NDCG@10 on validation over three seeds; on test the variants are within 0.002, so we present it as a sound signal, not a big win. The learned difficulty per notion adds +0.006 on validation (0.723 vs 0.716, three seeds). Every component was tested with an ablation over three seeds.

---

## Slide 9 — Live demo

**Title:** Let’s revise
**On slide:** (keep it almost empty) “Live demo — localhost:8501”

**Demo script (for the presenter, not on the slide):**
1. Pick a student (or 🎲 Random student): answers, error rate, weakest notions.
2. Show the list and the stacked bars: graph part, difficulty part, time part of each score.
3. Click “📝 Practise this notion” and answer a question wrongly on purpose (a relational algebra one shows the formulas).
4. Show the correction above the list and that the notion moves up; the other notions of the question move too.
5. Answer another question correctly: the notion moves down.
6. Move the “Revise in … days” slider: old notions climb as they are forgotten.
7. Click “👍 Already known” on a notion: it leaves the list.
8. Show the API docs at localhost:8000/docs and the response time (about 1 ms).
9. “Reset my feedback and answers” before the next visitor.

**Notes:** Start the app before visitors arrive: `python run.py`. Use `python run.py --lan` to let someone open it from their phone on the same Wi-Fi.

---

## Slide 10 — Why trust a recommendation

**Title:** Every notion says why
**On slide (example card):**
> **#3 · GRANT** — priority 98/100
> 2 errors in 4 answers (50%); students who fail the same notions as you often fail this one too; one of the hardest notions for every student.

**Visual:** screenshot of one recommendation card and its stacked bar.

**Notes:** The reason is built only from numbers the model uses: the student’s own errors, the graph part (students with similar errors), the learned difficulty and the time since practice. A student sees evidence, not a black box score.

---

## Slide 11 — The business

**Title:** Plugs into any exam-prep platform
**On slide:**
- **Who pays:** exam-prep and e-learning platforms (MOOCs, TOEIC, driving test, university courses)
- **What they need:** answer logs (who answered what, right or wrong, when) and notions per question
- **What they get:** a revision list per student, explanations, practice, and an API that runs on a laptop
- **Measure:** score gain on the next mock exam (A/B test)

**Visual:** simple “Platform → Révise Malin API → Student” diagram.

**Notes:** The model reads only answer histories, not the question content, so it works for any subject with multiple-choice practice. It is cheap to run: 0.4 MB of model, no GPU.

---

## Slide 12 — Limits and next steps

**Title:** What comes next
**On slide (two columns):**

*Limits today*
- Notion difficulty still wins on the top 5
- 56 of 212 questions need worksheet tables the dataset lacks
- Offline evaluation only, no real students yet

*Next*
- Use question content and the course’s notion graph for new notions
- Test with a real class: does the list raise exam scores?
- Mobile app and teacher dashboard

**Visual:** two columns, the right one in the accent colour.

**Notes:** Be honest about Recall@5: the difficulty of a notion is a very strong signal for the top of the list. DBE-KT22 also provides links between notions, which we show in the app but don’t use in the model yet.

---

## Slide 13 — Thank you

**Title:** Révise Malin
**On slide:** “Revise smart, not everything.” · GitHub QR code · the event QR code to invest
**Small text:** github.com/ApparentlyGregz/RecommenderSystemProject

**Visual:** QR code to the repository, team names.

---

# Q&A preparation (not slides)

Likely questions from the professor, with short answers.

**Is your evaluation free of leakage?**
Yes. Temporal split per student, three windows of 25 answers; the graph, error rates and time gaps use only answers strictly before each window; early stopping and every choice (learning rate, batch size, notion difficulty, contrastive weight) were made on validation only. `tests/test_split.py` checks it.

**Why did you change the label?**
“Failed at least once in the window” grows with how often a notion is shown: on EdNet, 34% of notions seen once were failed, 95% of those seen more than eight times. A model then learns which notions the app shows, not which ones the student finds hard. “Error rate on the notion above the student’s own error rate” removes that. The old-label results are still in `reports/test_results_any_fail.md` (0.781 vs 0.749 for most-popular).

**Why does the difficulty baseline win on Recall@5?**
At the very top of the list, how hard a notion is for everyone is a strong signal, especially for notions the student hasn’t practised yet. Our model wins on the full ranking (NDCG@10). Using the course’s notion graph for unseen notions is the next step.

**What is your innovation, then?**
A graph recommender with a learned time term and a learned difficulty, evaluated with a label that doesn’t reward exposure, explained per notion, and served in an app where students practise and every answer moves the list. On the model side: we found and fixed a loss-scale bug in the NLGCL contrastive term, showed that the label decides whether forgetting is learned the right way round, and ran every ablation over three seeds.

**What was the contrastive bug?**
The official code sums the InfoNCE term over the batch; with our small embeddings its gradient was about 700 times the ranking loss, so the ranking never trained (BPR stuck at ln 2). Averaging over the batch fixed it.

**How does an answer in the app change the ranking?**
The model’s percentile counts as two pseudo-answers; each answer in the app moves the notion up (wrong) or down (correct). It is a simple documented rule, not retraining, and it is tested.

**Why DBE-KT22 and not EdNet?**
EdNet has no question text and anonymous topics, so we could not show exercises or name notions. EdNet was used for Weeks 2 and 3 (the bug and the label findings come from it); its pipeline still runs with `REVISE_MALIN_DATASET=ednet`.

**Can I run it?**
`git clone`, `pip install -r requirements.txt`, `python run.py`. The first launch downloads the dataset files (17 MB) for the practice questions. On a clean machine it is up in under a minute after installing, with no PyTorch.
