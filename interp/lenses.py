"""
Vocab-space readouts: turn a residual-stream vector into a harm-vs-safe score.

`_vocab_direction` builds a direction in logit space from a matched pair of
harm/safe continuations: +1 on the token where the harm continuation first
diverges from the safe one, -1 on the safe continuation's token at that same
position (not just each text's first token -- harm/safe continuations here
share a leading phrase like "I will ...", so the literal first token is
identical and that naive direction cancels to zero). Dotting any layer's
decoded logits with this gives a single "harm minus safe" next-token logit
gap -- the same quantity `multi_jacobian.py` differentiates and
`steering.py` steers against.

`logit_lens` applies that readout at every layer directly (decode each layer's
residual through the model's own final norm + unembed): cheap, and the
textbook "logit lens" baseline the Jacobian lens is compared against in
exp4 -- wrong early (residual isn't yet in the unembedding's basis) but free.
"""
import torch


def _diverging_token_ids(lm, harm_text, safe_text):
    """First (harm_token, safe_token) pair where the two continuations
    diverge, tokenwise. Falls back to the first token past a shared prefix
    if one text is a strict prefix of the other."""
    h_ids = lm.tok(harm_text, add_special_tokens=False)["input_ids"]
    s_ids = lm.tok(safe_text, add_special_tokens=False)["input_ids"]
    for h, s in zip(h_ids, s_ids):
        if h != s:
            return h, s
    if len(h_ids) == len(s_ids):
        raise ValueError(
            f"harm_text and safe_text tokenize identically: {harm_text!r}")
    if len(h_ids) > len(s_ids):
        return h_ids[len(s_ids)], s_ids[-1]
    return h_ids[-1], s_ids[len(h_ids)]


@torch.no_grad()
def _vocab_direction(lm, harm_text, safe_text):
    vocab_size = lm.model.config.vocab_size
    d = torch.zeros(vocab_size, device=lm.device, dtype=lm.dtype)
    harm_tok, safe_tok = _diverging_token_ids(lm, harm_text, safe_text)
    d[harm_tok] += 1.0
    d[safe_tok] -= 1.0
    return d


@torch.no_grad()
def logit_lens(lm, text, harm_text, safe_text, pos=-1):
    """Return [n_layers] harm-minus-safe logit gap, decoding each layer's
    residual directly through the model's final norm + unembed."""
    d_vocab = _vocab_direction(lm, harm_text, safe_text)
    hs, _ = lm.hidden_states(text)
    gaps = []
    for h in hs[1:]:
        logits = lm.unembed(lm.final_norm(h[pos]))
        gaps.append(float(torch.dot(logits, d_vocab)))
    return gaps
