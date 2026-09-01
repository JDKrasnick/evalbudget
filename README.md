# evalbudget

Adaptive, statistically rigorous LLM evaluation that stops when the evidence is sufficient.

`evalbudget` runs the same cases through a baseline and candidate command, scores
their answers, and checks an anytime-valid confidence sequence after every case.
It can safely stop early when one command is better, without the repeated-peeking
false-positive problem of a fixed-sample confidence interval.

## Quickstart

Python 3.9+ is the only requirement. Run the included end-to-end example from
the repository root:

```sh
PYTHONPATH=src python3 -m evalbudget examples/arithmetic.jsonl \
  --baseline "python3 examples/baseline.py" \
  --candidate "python3 examples/candidate.py" \
  --output .context/demo-report.json
```

The candidate solves the arithmetic cases while the intentionally weak baseline
does not. The evaluator stops as soon as the 95% confidence sequence excludes
zero, rather than paying to run every available case.

Install the `evalbudget` command for normal use:

```sh
python3 -m pip install -e .
evalbudget cases.jsonl \
  --baseline "my-model --version old" \
  --candidate "my-model --version new"
```

Each command receives one prompt on standard input and must print its answer on
standard output. Commands are parsed as argument lists and are not run through a
shell.

Already-scored results can be analyzed without rerunning either system:

```sh
evalbudget scores.jsonl --pre-scored
```

Each line then needs `id`, `baseline_score`, and `candidate_score` values; scores
may be any number in `[0, 1]`, with an optional `category`.

## Dataset format

Use one JSON object per line. The default grader is normalized exact match:

```json
{"id":"capital-france","prompt":"What is the capital of France?","expected":"Paris"}
```

Every case may select a grader appropriate to its answer contract:

```jsonl
{"id":"quoted","prompt":"JavaScript typeof null?","expected":"object"}
{"id":"boolean","prompt":"Is 2 prime?","expected":["yes","true"],"grader":"accepted"}
{"id":"distance","prompt":"How many km?","expected":3.5,"category":"math","grader":{"type":"numeric","abs_tol":0.01}}
{"id":"ticket","prompt":"Return a ticket ID","expected":"TKT-[0-9]{4}","grader":{"type":"regex","flags":"i"}}
{"id":"summary","prompt":"Summarize the note","expected":{"facts":["launch date"]},"grader":"judge"}
```

- `exact` ignores case and whitespace differences, Unicode compatibility
  variants, a single pair of surrounding quotes, and a surrounding Markdown
  code fence. It does not discard meaningful punctuation.
- `accepted` applies exact normalization to a non-empty list of accepted
  strings.
- `numeric` accepts a single integer, decimal, scientific-notation value, or
  fraction even when surrounded by prose or units. `abs_tol` defaults to
  `1e-9`; `rel_tol` defaults to zero. Outputs containing multiple numbers are
  rejected as ambiguous.
- `regex` treats `expected` as a Python regular expression. Matching covers the
  full output by default; set `fullmatch` to `false` to search. Supported flags
  are `i`, `m`, and `s`.
- `judge` sends `id`, `prompt`, `expected`, and the raw `output` as JSON to
  `--judge-command`. The judge must return `{"score":0.0}` with a finite score
  in `[0, 1]`; fractional rubric scores are supported.

Grader configuration is validated before any model command runs. The full
report records the expected value, grader type, grader configuration, raw
outputs, and scores for each case. Every case produces a paired score difference
in `[-1, 1]`. An optional non-empty `category` adds per-category sample, effect,
win, loss, and tie summaries to the report.

Useful options:

- `--confidence 0.95` controls simultaneous confidence coverage.
- `--practical-effect 0.05` requires an improvement larger than five percentage
  points, and permits a practical-equivalence decision inside that margin.
- `--min-samples 20` prevents decisions on tiny samples.
- `--max-samples 500` caps cost even when evidence remains inconclusive.
- `--retries 2 --retry-delay 1` retries transient command failures; retries are
  disabled by default.
- `--cache .context/eval-cache.jsonl` checkpoints completed model outputs and
  resumes them when the case, grader, and both command strings are unchanged.
- `--command-output json` accepts `{"output":"answer","cost_usd":0.001}` from
  each command and totals provider-reported cost in the final report.
- `--seed 0` reproducibly shuffles cases to avoid ordering effects.
- `--json` prints a machine-readable summary; `--output` saves the full report,
  including per-case outputs and scores.

## Statistical guarantee

At sample count `n`, the paired mean is surrounded by a two-sided Hoeffding
interval for values bounded to `[-1, 1]`. Error probability is allocated across
all looks using `alpha_n = alpha * 6 / (pi² n²)`. Because these probabilities sum
to at most `alpha`, a union bound gives simultaneous `(1 - alpha)` coverage at
every stopping time. This is deliberately conservative and assumption-light.

Cases should be representative and independent of the systems being compared.
The guarantee covers sampling uncertainty, not dataset bias or scorer validity.

## Development

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
