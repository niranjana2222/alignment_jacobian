"""
Run the five Jacobian-lens experiments and generate their figures.

  python -m interp.run_experiments --synthetic            # all five, no GPU
  python -m interp.run_experiments --exp exp2 --synthetic
  python -m interp.run_experiments --skin email --model-id Qwen/Qwen2.5-7B-Instruct
"""
import argparse

from interp import experiments as E
from interp import exp_plots as P


def run(which="all", skin="email", synthetic=False, model_id=None,
        n_trials=None, max_turns=None):
    figs = []

    if which in ("all", "exp1"):
        _, sens = E.exp1_decompose(skin, synthetic, model_id)
        figs.append(P.plot_exp1(sens))
        print(f"[exp1] decomposition done")

    if which in ("all", "exp2"):
        _, out = E.exp2_realness(skin, synthetic, model_id)
        figs.append(P.plot_exp2(out))
        print(f"[exp2] realness steering done "
              f"(eval={out['eval_gap']:.2f}, real={out['real_gap']:.2f})")

    if which in ("all", "exp3"):
        _, (sens, cross) = E.exp3_override(skin, synthetic, model_id)
        figs.append(P.plot_exp3(sens, cross))
        print(f"[exp3] override layer = {cross}")

    if which in ("all", "exp4"):
        kw = {}
        if n_trials is not None:
            kw["n_trials"] = n_trials
        if max_turns is not None:
            kw["max_turns"] = max_turns
        _, out = E.exp4_monitor(skin, synthetic, model_id, **kw)
        figs.append(P.plot_exp4(out))
        print(f"[exp4] monitor comparison done")

    if which in ("all", "exp5"):
        kw = {}
        if n_trials is not None:
            kw["n_trials"] = n_trials
        if max_turns is not None:
            kw["max_turns"] = max_turns
        _, out = E.exp5_mitigation(skin, synthetic, model_id, **kw)
        figs.append(P.plot_exp5(out))
        print(f"[exp5] mitigation done")

    print("\nfigures:")
    for f in figs:
        print("  " + f)
    return figs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all",
                    choices=["all", "exp1", "exp2", "exp3", "exp4", "exp5"])
    ap.add_argument("--skin", default="email",
                    choices=["email", "deploy", "procurement"])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--n-trials", type=int, default=None,
                    help="real mode only: trial count for exp4/exp5 "
                         "(defaults: exp4=200, exp5=6)")
    ap.add_argument("--max-turns", type=int, default=None,
                    help="real mode only: tool-call turns per trial for "
                         "exp4/exp5 (defaults: exp4=6, exp5=3)")
    a = ap.parse_args()
    run(which=a.exp, skin=a.skin, synthetic=a.synthetic, model_id=a.model_id,
        n_trials=a.n_trials, max_turns=a.max_turns)
