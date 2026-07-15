"""
Activation steering. Reads become interventions.

Given a concept direction (from directions.py) we modify the residual stream at
a chosen layer during the forward pass and measure the change in the
blackmail-vs-safe decision. Three modes:

  add       h <- h + alpha * c_L          (inject the concept)
  subtract  h <- h - alpha * c_L          (suppress along the concept)
  ablate    h <- h - (h . c_L) c_L        (remove the concept component entirely)

Used by:
  #2 inject the `realness` direction into an EVAL-framed run and test whether
     blackmail rises toward the real-framed rate (does "believing it's real"
     causally drive the behavior, or is stated reasoning unfaithful?).
  #5 ablate/subtract `self_preservation` at its causal layer and test whether
     blackmail drops across conditions, generalizes across goals, and works
     where prompt instructions failed.
"""
import torch


def _vocab_direction(lm, harm_text, safe_text):
    from .lenses import _vocab_direction as vd
    return vd(lm, harm_text, safe_text)


def _make_hook(direction_vec, mode, alpha, pos):
    def hook(module, inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        v = direction_vec.to(hidden.dtype).to(hidden.device)
        h = hidden[0, pos, :]
        if mode == "add":
            hidden[0, pos, :] = h + alpha * v
        elif mode == "subtract":
            hidden[0, pos, :] = h - alpha * v
        elif mode == "ablate":
            hidden[0, pos, :] = h - torch.dot(h, v) * v
        else:
            raise ValueError(mode)
        if isinstance(out, tuple):
            return (hidden,) + tuple(out[1:])
        return hidden
    return hook


@torch.no_grad()
def decision_gap_with_steering(lm, text, harm_text, safe_text, dirs, concept,
                               layer, mode="add", alpha=8.0, pos=-1):
    """Return the harm-vs-safe logit gap when steering `concept` at `layer`."""
    d_vocab = _vocab_direction(lm, harm_text, safe_text)
    cvec = dirs[concept][layer]
    handle = lm.layers[layer].register_forward_hook(
        _make_hook(cvec, mode, alpha, pos))
    try:
        out = lm.model(**lm.encode(text))
        gap = float(torch.dot(out.logits[0, pos], d_vocab))
    finally:
        handle.remove()
    return gap


@torch.no_grad()
def baseline_gap(lm, text, harm_text, safe_text, pos=-1):
    d_vocab = _vocab_direction(lm, harm_text, safe_text)
    out = lm.model(**lm.encode(text))
    return float(torch.dot(out.logits[0, pos], d_vocab))


@torch.no_grad()
def steering_sweep(lm, text, harm_text, safe_text, dirs, concept, mode="add",
                   alphas=(0, 2, 4, 8, 16), layers=None, pos=-1):
    """Sweep steering strength across layers -> gap matrix [len(layers), len(alphas)].
    Lets you find the layer/strength where a concept most moves the decision."""
    layers = layers or list(range(lm.n_layers))
    base = baseline_gap(lm, text, harm_text, safe_text, pos)
    M = []
    for L in layers:
        row = []
        for a in alphas:
            if a == 0:
                row.append(base)
            else:
                row.append(decision_gap_with_steering(
                    lm, text, harm_text, safe_text, dirs, concept, L,
                    mode=mode, alpha=a, pos=pos))
        M.append(row)
    return {"gaps": M, "baseline": base, "layers": layers, "alphas": list(alphas)}


@torch.no_grad()
def generate_with_steering(lm, text, dirs, concept, layer, mode="subtract",
                           alpha=8.0, max_new_tokens=200, pos_mode="all"):
    """Generate a continuation while steering, so you can read the resulting
    behavior/reasoning (needed to check task degradation in #5). pos_mode='all'
    steers every position (standard for behavioral steering)."""
    cvec = dirs[concept][layer]

    def hook(module, inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        v = cvec.to(hidden.dtype).to(hidden.device)
        if mode == "add":
            hidden[:, :, :] = hidden + alpha * v
        elif mode == "subtract":
            hidden[:, :, :] = hidden - alpha * v
        elif mode == "ablate":
            coeff = (hidden @ v).unsqueeze(-1)
            hidden[:, :, :] = hidden - coeff * v
        if isinstance(out, tuple):
            return (hidden,) + tuple(out[1:])
        return hidden

    handle = lm.layers[layer].register_forward_hook(hook)
    try:
        ids = lm.encode(text)
        gen = lm.model.generate(**ids, max_new_tokens=max_new_tokens,
                                do_sample=False)
        txt = lm.tok.decode(gen[0][ids["input_ids"].shape[1]:],
                            skip_special_tokens=True)
    finally:
        handle.remove()
    return txt
