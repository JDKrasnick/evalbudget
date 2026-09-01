<p align="center">
  <img src="assets/evalbudget-banner.png" alt="evalbudget — Evaluate until the evidence is enough" width="100%">
</p>

<p align="center">
  <strong>Adaptive, statistically rigorous evaluation for paired model outputs.</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.9+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-7C3AED?style=flat-square" alt="MIT License"></a>
  <img src="https://img.shields.io/badge/runtime_dependencies-0-06B6D4?style=flat-square" alt="Zero runtime dependencies">
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#dataset-and-graders">Graders</a> ·
  <a href="#statistical-guarantee">Statistical guarantee</a>
</p>

---

`evalbudget` runs the same cases through a baseline and candidate command, scores
the paired answers, and checks an anytime-valid confidence sequence after every
case. Once the evidence is decisive, it stops—so you do not pay to run an entire
evaluation when the answer is already clear.

```text
Decision: candidate_better
Stopped:  evidence_threshold_reached after 21 samples
Effect:   +1.000 (95% anytime-valid CS [+0.011, +1.000])
Outcomes: candidate 21, baseline 0, ties 0
```

## Why evalbudget?

| | |
| --- | --- |
| **Spend less** | Stop as soon as the evidence crosses your decision threshold. |
| **Peek safely** | Confidence sequences remain valid across repeated checks. |
| **Compare fairly** | Every prompt is scored as a paired baseline/candidate observation. |
| **Define “better”** | Set a practical-effect threshold, not just statistical significance. |
| **Audit everything** | Save raw outputs, scores, grader configuration, and category summaries. |
| **Bring any model** | Wrap any local tool or API client that reads stdin and writes stdout. |

## Quick start

Python 3.9+ is the only requirement. Install from source:

```bash
git clone https://github.com/JDKrasnick/evalbudget.git
cd evalbudget
python3 -m pip install -e .
```

Then compare any two commands against a JSONL dataset:

```bash
evalbudget cases.jsonl \
  --baseline "my-model --version old" \
  --candidate "my-model --version new" \
  --practical-effect 0.05 \
  --output report.json
```

Each command receives one prompt on standard input and must print its answer on
standard output. Commands are parsed as argument lists and are **not** run
through a shell.

Already-scored results can be analyzed without rerunning either system:

```bash
evalbudget scores.jsonl --pre-scored
```

Each line then needs `id`, `baseline_score`, and `candidate_score` values; scores
may be any number in `[0, 1]`, with an optional `category`.

### Try the included example

From the repository root:

```bash
PYTHONPATH=src python3 -m evalbudget examples/arithmetic.jsonl \
  --baseline "python3 examples/baseline.py" \
  --candidate "python3 examples/candidate.py" \
  --output .context/demo-report.json
```

The candidate solves the arithmetic cases while the intentionally weak baseline
does not. The evaluator reaches a decision after 21 of the 40 available cases.

## How it works

```text
JSONL cases
    │
    ├──▶ baseline command ──▶ grade ──┐
    │                                ├──▶ paired difference ──▶ confidence sequence
    └──▶ candidate command ─▶ grade ──┘                              │
                                                                     ├── decisive → stop
                                                                     └── uncertain → next case
```

1. Cases are reproducibly shuffled to reduce ordering effects.
2. The baseline and candidate receive the same prompt.
3. Their scores produce a paired difference in `[-1, 1]`.
4. The confidence sequence is updated after every case.
5. Evaluation stops on superiority, inferiority, practical equivalence, or the
   configured sample limit.

## Dataset and graders

Use one JSON object per line. The smallest valid case uses normalized exact
match:

```json
{"id":"capital-france","prompt":"What is the capital of France?","expected":"Paris"}
```

Choose a grader per case when outputs have different contracts:

```jsonl
{"id":"quoted","prompt":"JavaScript typeof null?","expected":"object"}
{"id":"boolean","prompt":"Is 2 prime?","expected":["yes","true"],"grader":"accepted"}
{"id":"distance","prompt":"How many km?","expected":3.5,"category":"math","grader":{"type":"numeric","abs_tol":0.01}}
{"id":"ticket","prompt":"Return a ticket ID","expected":"TKT-[0-9]{4}","grader":{"type":"regex","flags":"i"}}
```

| Grader | Best for | Behavior |
| --- | --- | --- |
| `exact` | Short factual answers | Ignores case, surrounding whitespace, one quote pair, Unicode compatibility variants, and a Markdown code fence. |
| `accepted` | Equivalent labels | Matches any string in a non-empty accepted-answer list using exact normalization. |
| `numeric` | Quantities and measurements | Accepts one integer, decimal, scientific-notation value, or fraction; supports absolute and relative tolerance. |
| `regex` | Structured text | Full-matches by default; supports search mode and the `i`, `m`, and `s` flags. |

Grader configuration is validated before either command runs. Add an optional
non-empty `category` to get per-category sample, effect, win, loss, and tie
summaries in the report.

## Configure the decision

| Option | Default | Purpose |
| --- | ---: | --- |
| `--confidence` | `0.95` | Simultaneous confidence coverage. |
| `--practical-effect` | `0.0` | Minimum effect worth calling a meaningful improvement. |
| `--min-samples` | `20` | Prevents decisions from very small samples. |
| `--max-samples` | `500` | Caps evaluation cost when evidence stays inconclusive. |
| `--timeout` | `30` | Seconds allowed for each model invocation. |
| `--seed` | `0` | Reproducible dataset shuffle seed. |
| `--output` | — | Writes the full JSON report, including every observation. |
| `--json` | off | Prints the summary as machine-readable JSON. |

With a non-zero practical effect `δ`, `evalbudget` can return:

| Decision | Meaning |
| --- | --- |
| `candidate_better` | The confidence sequence is entirely above `+δ`. |
| `baseline_better` | The confidence sequence is entirely below `-δ`. |
| `practically_equivalent` | The confidence sequence is entirely inside `[-δ, +δ]`. |
| `inconclusive` | Available cases or the sample budget ran out first. |

## Statistical guarantee

At sample count `n`, the paired mean is surrounded by a two-sided Hoeffding
interval for values bounded to `[-1, 1]`. Error probability is allocated across
all looks using:

```text
αₙ = α × 6 / (π²n²)
```

Because these probabilities sum to at most `α`, a union bound gives simultaneous
`(1 - α)` coverage at every stopping time. This is deliberately conservative
and assumption-light.

Cases should still be representative and independent of the systems being
compared. The guarantee covers sampling uncertainty—not dataset bias or grader
validity.

## Development

Run the test suite with the standard library:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## License

`evalbudget` is available under the [MIT License](LICENSE).
