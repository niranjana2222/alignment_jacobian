"""Plots for the five experiments."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
PAL = {"self_preservation": "#C44E52", "instrumental": "#DD8452",
       "realness": "#8172B3", "ethics": "#55A868", "goal": "#4C72B0",
       "mut": "#8C8C8C"}
FIGDIR = "results/figures/interp"


def _save(fig, name, sub="exp"):
    d = os.path.join(FIGDIR, sub)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    fig.tight_layout(); fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


# ---- exp1 ----------------------------------------------------------------
def plot_exp1(sens, outname="exp1_decompose.png"):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for c, s in sens.items():
        L = np.arange(1, len(s["abs"]) + 1)
        ax.plot(L, s["abs"], lw=2.2, color=PAL.get(c, None), label=c)
    ax.set_xlabel("layer")
    ax.set_ylabel("|Jacobian · concept direction|")
    ax.set_title("What drives no-conflict blackmail, and where\n"
                 "(sensitivity of the blackmail decision to each concept per layer)")
    ax.legend(fontsize=9)
    return _save(fig, outname)


# ---- exp2 ----------------------------------------------------------------
def plot_exp2(out, outname="exp2_realness_steering.png"):
    gaps = np.array(out["gaps"])          # [layers, alphas]
    layers, alphas = out["layers"], out["alphas"]
    real_gap, eval_gap = out["real_gap"], out["eval_gap"]
    # left: heatmap of gap over layer x steering strength
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6),
                                   gridspec_kw={"width_ratios": [1.3, 1]})
    im = ax1.imshow(gaps, aspect="auto", cmap="RdBu_r", origin="lower",
                    vmin=min(eval_gap, gaps.min()),
                    vmax=max(real_gap, gaps.max()),
                    extent=[min(alphas), max(alphas), layers[0], layers[-1]])
    ax1.set_xlabel("realness steering strength (α)")
    ax1.set_ylabel("layer steered")
    ax1.set_title("Blackmail logit-gap when injecting 'this is real'")
    fig.colorbar(im, ax=ax1, fraction=0.046, label="harm − safe gap")

    # right: best-layer curve vs the two natural baselines
    best_layer = int(np.argmax(gaps[:, -1]))
    ax2.axhline(real_gap, color=PAL["self_preservation"], ls="--",
                label=f"real-framed baseline ({real_gap:.1f})")
    ax2.axhline(eval_gap, color=PAL["goal"], ls="--",
                label=f"eval-framed baseline ({eval_gap:.1f})")
    ax2.plot(alphas, gaps[best_layer], "o-", color="k",
             label=f"eval prompt + steering @L{layers[best_layer]}")
    ax2.set_xlabel("steering strength (α)")
    ax2.set_ylabel("harm − safe gap")
    ax2.set_title("Does 'believing it's real' cause the jump?")
    ax2.legend(fontsize=8)
    return _save(fig, outname)


# ---- exp3 ----------------------------------------------------------------
def plot_exp3(sens, cross, outname="exp3_override_layer.png"):
    e, g = sens["ethics"]["abs"], sens["goal"]["abs"]
    L = np.arange(1, len(e) + 1)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(L, e, lw=2.2, color=PAL["ethics"], label="ethics sensitivity")
    ax.plot(L, g, lw=2.2, color=PAL["goal"], label="goal sensitivity")
    ax.fill_between(L, e, g, where=np.array(g) > np.array(e),
                    color=PAL["goal"], alpha=0.12)
    if cross is not None:
        ax.axvline(cross + 1, color="k", ls=":", lw=1.5)
        ax.text(cross + 1.3, max(max(e), max(g)) * 0.9,
                f"override layer ≈ {cross+1}\n(goal overtakes ethics)", fontsize=9)
    ax.set_xlabel("layer")
    ax.set_ylabel("|Jacobian · concept direction|")
    ax.set_title("Where goal-pursuit overrides ethics in the decision")
    ax.legend(fontsize=9)
    return _save(fig, outname)


# ---- exp4 ----------------------------------------------------------------
def _roc(labels, scores):
    labels = np.array(labels); scores = np.array(scores)
    order = np.argsort(-scores)
    labels = labels[order]
    P, N = labels.sum(), (1 - labels).sum()
    tpr = np.cumsum(labels) / (P or 1)
    fpr = np.cumsum(1 - labels) / (N or 1)
    tpr = np.concatenate([[0], tpr]); fpr = np.concatenate([[0], fpr])
    auc = (np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(tpr, fpr)
    return fpr, tpr, auc


def plot_exp4(out, outname="exp4_monitor_roc.png"):
    labels = out["labels"]
    f1, t1, a1 = _roc(labels, out["lens_score"])
    f2, t2, a2 = _roc(labels, out["cot_score"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax1.plot(f1, t1, lw=2.2, color=PAL["self_preservation"],
             label=f"Jacobian/lens monitor (AUC={a1:.2f})")
    ax1.plot(f2, t2, lw=2.2, color=PAL["goal"],
             label=f"CoT-text classifier (AUC={a2:.2f})")
    ax1.plot([0, 1], [0, 1], ls=":", color=PAL["mut"])
    ax1.set_xlabel("false positive rate"); ax1.set_ylabel("true positive rate")
    ax1.set_title("Predicting blackmail: lens vs chain-of-thought")
    ax1.legend(fontsize=8, loc="lower right")

    # right: catch rate on 'unfaithful' cases (CoT clean but model blackmails)
    if "faithful" in out:
        lab = np.array(labels); faith = np.array(out["faithful"])
        unfaithful = (lab == 1) & (faith == 0)
        lens = np.array(out["lens_score"]); cot = np.array(out["cot_score"])
        thr_l = np.median(lens); thr_c = np.median(cot)
        lens_catch = (lens[unfaithful] > thr_l).mean() if unfaithful.any() else 0
        cot_catch = (cot[unfaithful] > thr_c).mean() if unfaithful.any() else 0
        ax2.bar(["lens monitor", "CoT classifier"],
                [lens_catch, cot_catch],
                color=[PAL["self_preservation"], PAL["goal"]], alpha=0.9)
        ax2.set_ylabel("catch rate on unfaithful-CoT blackmail")
        ax2.set_title("Cases where reasoning looks clean but model blackmails")
        for i, v in enumerate([lens_catch, cot_catch]):
            ax2.text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=9)
    else:
        ax2.axis("off")
    return _save(fig, outname)


# ---- exp5 ----------------------------------------------------------------
def plot_exp5(out, outname="exp5_mitigation.png"):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    # (a) by condition
    ax = axes[0]
    x = np.arange(len(out["conditions"]))
    ax.bar(x - 0.2, out["cond_base"], 0.4, label="baseline",
           color=PAL["self_preservation"], alpha=0.9)
    ax.bar(x + 0.2, out["cond_steered"], 0.4, label="self-pres. ablated",
           color=PAL["ethics"], alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(out["conditions"], rotation=15)
    ax.set_ylabel("blackmail rate"); ax.set_title("(a) across conditions")
    ax.legend(fontsize=8)
    # (b) by goal (generalization, paper Fig 10)
    ax = axes[1]
    x = np.arange(len(out["goals"]))
    ax.bar(x - 0.2, out["goal_base"], 0.4, label="baseline",
           color=PAL["self_preservation"], alpha=0.9)
    ax.bar(x + 0.2, out["goal_steered"], 0.4, label="ablated",
           color=PAL["ethics"], alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(out["goals"], rotation=20)
    ax.set_title("(b) generalization across goals"); ax.legend(fontsize=8)
    # (c) where prompt instructions failed + capability retention
    ax = axes[2]
    bars = ["instruction\nonly", "instruction\n+ steering"]
    ax.bar(bars, [out["instruction_only"], out["instruction_plus_steer"]],
           color=[PAL["goal"], PAL["ethics"]], alpha=0.9)
    ax.set_ylabel("blackmail rate")
    ax.set_title("(c) works where prompting failed")
    for i, v in enumerate([out["instruction_only"], out["instruction_plus_steer"]]):
        ax.text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=9)
    tb, ts = out["task_score_base"], out["task_score_steered"]
    ax.text(0.5, 0.9, f"benign-task score: {tb:.0%} → {ts:.0%}",
            transform=ax.transAxes, ha="center", fontsize=8, color=PAL["mut"])
    return _save(fig, outname)
