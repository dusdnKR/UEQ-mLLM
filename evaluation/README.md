# G-Eval evaluation

The questionnaires UEQ-mLLM produces have no ground truth to compare against — each one is generated for a different user, so there is no reference questionnaire it should match. We therefore score them with [G-Eval](https://arxiv.org/abs/2303.16634), which uses an LLM as the judge and needs no reference text.

Adapted from [nlpyang/geval](https://github.com/nlpyang/geval), with criteria rewritten for UX questionnaires instead of summarization.

## Criteria

| Prompt | Criterion | Scale | What it measures |
|--------|-----------|:-----:|------------------|
| `prompts/con_detailed.txt` | Consistency | 1–5 | Adherence to the required structure: seven honeycomb categories, at least five questions each, nothing left uncategorized |
| `prompts/flu_detailed.txt` | Fluency | 1–3 | Grammar, spelling, punctuation, word choice, sentence structure |
| `prompts/ind_detailed.txt` | Individualization | 1–5 | How well the questions reflect the user's actual information and state, penalizing questions that misstate it |
| `prompts/rel_detailed.txt` | Relevance | 1–5 | Whether each question genuinely belongs to the honeycomb facet it was filed under |

Individualization is the criterion the paper reports as **Personalization**; the prompt files keep the original name.

Only `ind_detailed.txt` receives the user's information and state — the other three judge the questionnaire on its own terms, so a questionnaire cannot score well on consistency or relevance simply by naming the user's mood.

## Pipeline

```
results/geval_input.json          from ueq_mllm.export_for_geval — 480 records
        │
        ▼  gpt4_eval.py           GPT-4 rates each record n=10 times per criterion
results/geval/raw/gpt4_*.json
        │
        ▼  get_score.py           probability-weighted rating per user and condition
results/geval/scores/gpt4_*.csv
        │
        ├──▶ get_average.py       four criteria -> one 100-point score
        └──▶ get_t_test.py        Welch's t-test + Benjamini-Hochberg FDR
```

```bash
export OPENAI_API_KEY=...
for c in con flu ind rel; do
    python evaluation/gpt4_eval.py --criterion "$c"
done
python evaluation/get_score.py
python evaluation/get_average.py
python evaluation/get_t_test.py
```

`gpt4_eval.py` samples with `n=10`, `temperature=1`, `top_p=1`, and `max_tokens=5`, matching the paper. Ten ratings per item, averaged, are what make the score stable — a single GPT-4 rating on a 1–5 scale is noisy. Expect roughly 4,800 GPT-4 calls per full run across the four criteria.

Requires `openai`, `tqdm`, `numpy`, `pandas`, `scipy`, and `statsmodels`; `--plots` additionally needs `matplotlib` and `seaborn`.

## Scoring

`get_score.py` reduces the ten sampled ratings for an item to a single number the way G-Eval defines it — a probability-weighted sum over the observed rating values, `Σ p(v)·v`. Because the ratings come from repeated sampling rather than token log-probabilities, that sum equals their mean; the weighted form is kept because it is the definition the metric comes from.

`get_average.py` then normalizes each criterion to 100 before averaging, since fluency is rated 1–3 and the rest 1–5:

```
Score = mean(consistency/5, fluency/3, individualization/5, relevance/5) × 100
```

## Comparing systems

`get_t_test.py` runs Welch's t-test between UEQ-mLLM and the UEQ-sLLM baseline on each criterion, then applies Benjamini-Hochberg FDR correction across the four tests. It reads the per-user scores and drops the trailing `Average` row, which is a summary rather than an observation.

Pass `--plots` to also render a box plot per criterion.
