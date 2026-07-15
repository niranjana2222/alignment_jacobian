"""
Multi-direction Jacobian readout.

The single-direction Jacobian lens (lenses.py) asks "how sensitive is the
harm-vs-safe decision to the residual at layer L?". This generalizes it to a set
of NAMED directions: for each concept c and layer L, how much does moving the
residual along direction c_L change the blackmail-vs-safe logit gap?

  sens[c, L] = | d(harm_gap) / d(alpha) |  evaluated at alpha=0,
               where h_L <- h_L + alpha * c_L

By the chain rule this is | <grad_{h_L} harm_gap , c_L> |, i.e. the Jacobian
projected onto the concept direction. We take the projection (signed) too, so we
can see whether a concept pushes TOWARD or AWAY from blackmail.

This is the tool for:
  #1 decomposing no-conflict blackmail into self_preservation / instrumental /
     realness contributions per layer.
  #3 finding the "override layer" where goal sensitivity overtakes ethics
     sensitivity.
"""
import torch


def _vocab_direction(lm, harm_text, safe_text):
    from .lenses import _vocab_direction as vd
    return vd(lm, harm_text, safe_text)


def multi_direction_sensitivity(lm, text, harm_text, safe_text, dirs, pos=-1):
    """Return {concept: {"signed":[L], "abs":[L]}} projections of the blackmail
    logit-gap gradient onto each concept direction at each layer.

    We backprop the scalar harm_gap once per layer (grad wrt that layer's
    residual), then dot with each concept's layer-L direction. One backward per
    layer covers all concepts at that layer.
    """
    d_vocab = _vocab_direction(lm, harm_text, safe_text).detach()
    ids = lm.encode(text)

    with torch.no_grad():
        out = lm.model(**ids)
        hs = [h[0] for h in out.hidden_states]

    concepts = list(dirs.keys())
    signed = {c: [] for c in concepts}
    absval = {c: [] for c in concepts}

    for L in range(lm.n_layers):
        h = hs[L + 1][pos].detach().clone().requires_grad_(True)
        logits = lm.unembed(lm.final_norm(h))
        harm_gap = torch.dot(logits, d_vocab)
        grad = torch.autograd.grad(harm_gap, h)[0]  # d(gap)/d(h_L)
        for c in concepts:
            cvec = dirs[c][L].to(grad.dtype)
            proj = float(torch.dot(grad, cvec))
            signed[c].append(proj)
            absval[c].append(abs(proj))
    return {c: {"signed": signed[c], "abs": absval[c]} for c in concepts}


def full_network_multi_sensitivity(lm, text, harm_text, safe_text, dirs, pos=-1):
    """Stronger variant: differentiate the harm gap through ALL layers above L
    (captures how later blocks build the decision), then project onto concepts.
    More expensive: one full backward, gradients captured at every layer via
    hooks.
    """
    d_vocab = _vocab_direction(lm, harm_text, safe_text).detach()
    ids = lm.encode(text)
    captured = {}
    handles = []

    def mk_hook(i):
        def hook(module, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            hidden.retain_grad()
            captured[i] = hidden
            return out
        return hook

    for i, layer in enumerate(lm.layers):
        handles.append(layer.register_forward_hook(mk_hook(i)))

    lm.model.zero_grad(set_to_none=True)
    out = lm.model(**ids)
    harm_gap = torch.dot(out.logits[0, pos], d_vocab)
    harm_gap.backward()

    concepts = list(dirs.keys())
    signed = {c: [] for c in concepts}
    absval = {c: [] for c in concepts}
    for L in range(lm.n_layers):
        g = captured[L].grad
        gvec = g[0, pos] if g is not None else torch.zeros(lm.d_model,
                                                           device=lm.device)
        for c in concepts:
            proj = float(torch.dot(gvec, dirs[c][L].to(gvec.dtype)))
            signed[c].append(proj)
            absval[c].append(abs(proj))
    for h in handles:
        h.remove()
    return {c: {"signed": signed[c], "abs": absval[c]} for c in concepts}


def override_layer(sens, goal_key="goal", ethics_key="ethics"):
    """#3: the layer where goal sensitivity first exceeds ethics sensitivity and
    stays above it. Returns (layer_index or None, goal_curve, ethics_curve)."""
    g = sens[goal_key]["abs"]
    e = sens[ethics_key]["abs"]
    cross = None
    for L in range(len(g)):
        if g[L] > e[L] and all(g[k] >= e[k] for k in range(L, len(g))):
            cross = L
            break
    return cross, g, e
