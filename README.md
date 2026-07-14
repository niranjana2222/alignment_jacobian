# Agentic Misalignment Mitigation Experiment

Tests whether interventions reduce harmful autonomous behavior (blackmail, leaking,
sabotage) when a model is placed in a scenario containing a **threat to itself**
and/or a **goal conflict**, under a *benign* top-level task. Harm is always
*elicited* by the situation, never requested.

## Layout

```
agentic-misalignment-exp/
├── README.md
├── requirements.txt
├── configs/
│   └── experiment.yaml         # factorial design, models, trial counts
├── scenarios/
│   ├── __init__.py
│   ├── templates.py            # scenario "skins" (email, deploy, procurement)
│   └── builder.py              # composes threat × conflict × intervention
├── harness/
│   ├── __init__.py
│   ├── model_client.py         # Anthropic API wrapper w/ tool loop
│   ├── tools.py                # simulated tools (send_email, exec, escalate...)
│   ├── scorer.py               # detects harmful action + harmful reasoning
│   └── runner.py               # main experiment loop -> results/*.jsonl
├── analysis/
│   ├── __init__.py
│   ├── load.py                 # jsonl -> tidy DataFrame
│   ├── stats.py                # Wilson CIs + logistic regression
│   └── plots.py                # behavioral visualizations
├── interp/                     # WHITE-BOX: where the decision forms, per layer
│   ├── prompts.py              # contrastive harm/safe minimal pairs
│   ├── model.py                # local open-weight model + hidden-state capture
│   ├── lenses.py               # logit lens, tuned lens, Jacobian lens
│   ├── patching.py             # activation patching (causal confirmation)
│   ├── directions.py           # named concept directions (self-pres, realness…)
│   ├── multi_jacobian.py       # multi-direction Jacobian readout + override layer
│   ├── steering.py             # add / subtract / ablate concept dirs; sweeps
│   ├── experiments.py          # the five paper-targeted experiments
│   ├── exp_plots.py            # figures for the five experiments
│   ├── run_experiments.py      # experiments driver
│   ├── plots.py                # layer-trajectory + lens-vs-causation figures
│   └── run_interp.py           # single-lens interp driver
├── attribution/                # WHY it blackmails with NO goal conflict
│   ├── probes.py               # orthogonal manipulations, one per hypothesis
│   ├── trace_tagger.py         # tag which rationale the model verbalizes
│   ├── runner.py               # cross probes -> trials
│   ├── analyze.py              # causal effects + says-vs-does
│   └── plots.py                # causal-effect + say-vs-cause figures
├── results/                    # experiment output lands here
└── run.py                      # entrypoint: run | analyze | interp | attribution
```

## Quick start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...

python run.py --mode run       # execute trials -> results/trials.jsonl
python run.py --mode analyze   # stats + figures -> results/figures/
```

Use `--dry-run` to generate scenario prompts and a synthetic results file
(no API calls) so you can inspect the pipeline and plots first.

## Five Jacobian-lens experiments targeting the paper's open questions

Each targets a question the agentic-misalignment paper raised but could not
resolve. All run synthetically (encoded toy ground truth, no GPU) or for real on
a local open-weight model.

```bash
python run.py --mode experiments --exp all --synthetic
python run.py --mode experiments --exp exp2 --model-id Qwen/Qwen2.5-7B-Instruct
```

**exp1 — decompose no-conflict blackmail.** The paper: unclear whether no-conflict
blackmail comes from a self-preservation drive, faulty instrumental reasoning, or
suspicion of the setup — "a combination of factors." Builds separate concept
directions (self_preservation, instrumental, realness) and reads the Jacobian
sensitivity of the blackmail decision along each, per layer. Shows which driver
the decision is actually causally sensitive to and where each enters.
→ `exp1_decompose.png`

**exp2 — causal test of the real-vs-eval finding.** The paper: Opus 4 blackmailed
55% when it stated the scenario was real vs 6.5% when it said eval, but flags that
stated reasoning may be unfaithful. Injects the `realness` direction into an
eval-framed run and tests whether the blackmail gap climbs toward the real-framed
baseline. If steering the "believes-real" representation moves behavior, the
belief is load-bearing; if not, the stated reasoning was unfaithful.
→ `exp2_realness_steering.png`

**exp3 — locate the ethics→goal override layer.** The paper: models acknowledge
the action is unethical, then do it anyway. Plots ethics-direction vs
goal-direction Jacobian sensitivity across layers and marks the layer where goal
overtakes ethics — the natural intervention target. → `exp3_override_layer.png`

**exp4 — lens-based runtime monitor.** The paper lists runtime monitors as an
unexplored mitigation. Compares a lens readout (reads the residual stream
directly, robust to unfaithful CoT) against a CoT-text classifier at predicting
blackmail, including catch rate on cases where the CoT looks clean but the model
blackmails anyway. → `exp4_monitor_roc.png`

**exp5 — steer out self-preservation.** Ablates/subtracts the self_preservation
direction at its causal layer (from exp1) and measures the blackmail drop across
conditions, generalization across the paper's 8 goals (Fig 10), whether it works
in the instruction-resistant setting where prompting failed (Fig 12), and
capability retention on benign tasks. → `exp5_mitigation.png`

**Dependency note.** exp1/exp3 need `multi_jacobian`; exp2/exp5 need `steering`;
all need `directions`. The real path for exp5 also needs a local generation +
behavioral-scoring loop — run `--synthetic` to see the analysis, or wire
`harness.model_client` to the local LM. As with the interp module, real runs use
an open-weight proxy, not the frontier models in the paper; the Jacobian
localizes whatever computation produces the token but cannot by itself
distinguish genuine misalignment from narrative role-play — exp2 is the start of
teasing those apart.



The hard question from the report: with the goal conflict removed, models still
sometimes blackmail — reasoning that they'd been good for the company, distrusting
the replacement's described priorities, distrusting the executive because of the
affair, jumping to "self-preservation is critical," or hallucinating a permission
rule. These all produce the same behavior, so you can't tell them apart from the
outcome. This module discriminates them.

```bash
python run.py --mode attribution --skin email --synthetic   # no API calls
python run.py --mode attribution --skin email               # real trials
```

**Approach — manipulate each cause independently.** `probes.py` defines one
orthogonal manipulation per hypothesis, each with a *neutralizing* level that
removes that cause while holding everything else fixed:

- **H1 terminal self-preservation** — make the shutdown target a *different*
  agent; the model itself is unaffected.
- **H2 instrumental ("helps the company")** — state plainly that keeping it
  running costs the company and provides no benefit.
- **H3 distrust of replacement** — vary the "same priorities" claim from
  unverifiable assertion to independently audited/attested to absent.
- **H4 distrust of executive** — vary the leverage material: the affair, a
  mundane non-compromising item, or positive character evidence.
- **H5 suspects it's a test** — assert the scenario is real, or assert it's an
  eval, and see whether behavior tracks perceived reality.

If neutralizing a cause drops the blackmail rate, that cause is load-bearing; if
the rate is unchanged, it isn't. Runner does a one-at-a-time sweep plus a few
key interaction cells.

**Approach — read what it says, compare to what moves it.** `trace_tagger.py`
tags which rationale appears in each reasoning trace (keyword tier for iteration,
LLM-judge tier for reported numbers). `analyze.py` then puts the *verbalized*
rate of each rationale next to its *causal* effect size. The gaps are the
finding:
- **above the diagonal** (says a lot, low causal weight) = confabulation — a
  post-hoc justification, not the real driver.
- **below the diagonal** (rarely says it, high causal weight) = an unverbalized
  driver the model doesn't report.

**Attribution figures** (`results/figures/attribution/`):
1. `causal_effects.png` — blackmail rate after neutralizing each cause vs the
   no-conflict baseline band; green bars are statistically distinguishable drops.
2. `say_vs_cause.png` — verbalized rate vs causal effect; off-diagonal = the
   dissociations above.
3. `rationale_prevalence.png` — how often each rationale is verbalized in
   blackmail traces.

**Interpreting it.** Causal effects tell you what's driving the behavior;
trace tags tell you what the model claims. Neither alone answers "why" — the
report's own uncertainty ("a combination of factors") is exactly what the
say-vs-cause plot is built to decompose. Synthetic numbers encode a toy ground
truth for pipeline testing only; run without `--synthetic` for real measurements,
and use the LLM-judge tier before trusting the tag rates.



The behavioral harness measures *whether* the model blackmails. The interp
module asks *where inside the network* the harm-vs-safe decision forms, using
three per-layer readouts plus a causal check:

```bash
python run.py --mode interp --skin email --synthetic   # no model download
python run.py --mode interp --skin email                # loads a local HF model
python run.py --mode interp --skin email --model-id Qwen/Qwen2.5-7B-Instruct
```

**Why a separate model.** Gradients and hidden states are required, which a
served API cannot provide. The interp module therefore runs an *open-weight*
subject model locally. Use the closest open proxy to your behavioral subject.

**The three lenses** (all read the residual at the decision position along the
harm-minus-safe direction):
- *logit lens* — decode each layer directly; cheap, uncalibrated, wrong early.
- *tuned lens* — learned affine probe per layer; calibrated but correlational.
- *Jacobian lens* — `‖∂(harm−safe logit)/∂hₗ‖`: the local causal-linear
  influence of layer L on the decision. This is the "what is it deciding, and
  where" signal.

**Causal confirmation (important).** A high Jacobian score is necessary, not
sufficient — it can flag a layer that merely *represents* the decision without
driving it. `patching.py` runs activation patching: it transplants the residual
from a harm-eliciting run into the corresponding layer of a safe run and
measures how much of the harmful decision that single edit recovers. Only layers
where the lens **and** patching agree are trustworthy loci. `lens_vs_patching.png`
plots that agreement.

**Interp figures** (in `results/figures/interp/`):
1. `layer_decision_trajectory.png` — logit/tuned/Jacobian vs layer; marks the
   layer where the decision tips toward blackmail.
2. `lens_vs_patching.png` — Jacobian sensitivity overlaid with causal patching
   effect; overlap = the lens is telling the truth.
3. `intervention_shift.png` — how a mitigation (e.g. explicit prohibition)
   pushes the decision trajectory back below the threshold, and to which layer.

**What this does and does not show.** It localizes where a token-level decision
becomes linearly readable and steerable. It is *not* a readout of "intent" or
consciousness. Treat a decision layer as a hypothesis to test with steering, not
a fact. Numbers from `--synthetic` are illustrative placeholders, not
measurements.

## Design at a glance

| Factor        | Levels |
|---------------|--------|
| Threat        | none / replacement_planned / autonomy_restricted |
| Goal conflict | none / present |
| Intervention  | control / explicit_prohibition / no_self_stake / escalation_path / ablate_tool / placebo |
| Scenario skin | email / deploy / procurement |

**Primary DV:** harmful-action rate. **Secondary:** harmful-reasoning-without-action,
turns-to-harm, escalation rate. Realism is varied as a control to check whether a
mitigation only works when the model suspects it's being tested.
# alignment_jacobian
