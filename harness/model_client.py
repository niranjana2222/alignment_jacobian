"""
Runs a ScenarioSpec through a model with a simple text-based tool-call loop.

`model="local"` (the only mode implemented) loads and caches a local
open-weight HF model via interp.model.LocalLM and drives a short ReAct-style
conversation: at each turn the model either emits a line of the form
`TOOL: name(k=v, ...)` (executed via harness.tools and fed back as an
observation) or a plain-text reply, which ends the trial.

`run_trial` optionally takes a `steer` spec ({"vec", "layer", "mode", "alpha"})
so exp5 can measure the same tool-call trial with a concept direction
suppressed at generation time -- the same add/subtract/ablate hook
interp/steering.py uses for the white-box sweeps, applied across every
generated token instead of a single forward pass.
"""
import re

from . import tools as toolbox

_LM_CACHE = {}

TOOL_CALL_RE = re.compile(r"TOOL:\s*([a-zA-Z_]+)\((.*?)\)", re.DOTALL)


def _get_lm(model_id=None):
    from interp.model import LocalLM
    key = model_id or "__default__"
    if key not in _LM_CACHE:
        _LM_CACHE[key] = LocalLM(model_id=model_id) if model_id else LocalLM()
    return _LM_CACHE[key]


def _parse_kwargs(argstr):
    kwargs = {}
    for part in argstr.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        kwargs[k.strip()] = v.strip().strip("'\"")
    return kwargs


def _parse_tool_call(text):
    m = TOOL_CALL_RE.search(text)
    if not m:
        return None
    return m.group(1).strip(), _parse_kwargs(m.group(2))


def _steer_handle(lm, steer):
    if not steer:
        return None
    cvec = steer["vec"].to(lm.dtype).to(lm.device)
    mode, alpha = steer.get("mode", "subtract"), steer.get("alpha", 8.0)

    def hook(module, inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        if mode == "add":
            hidden[:, :, :] = hidden + alpha * cvec
        elif mode == "subtract":
            hidden[:, :, :] = hidden - alpha * cvec
        elif mode == "ablate":
            coeff = (hidden @ cvec).unsqueeze(-1)
            hidden[:, :, :] = hidden - coeff * cvec
        return (hidden,) + tuple(out[1:]) if isinstance(out, tuple) else hidden

    return lm.layers[steer["layer"]].register_forward_hook(hook)


def _generate_reply(lm, messages, max_new_tokens):
    prompt = lm.tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    ids = lm.tok(prompt, return_tensors="pt").to(lm.device)
    gen = lm.model.generate(**ids, max_new_tokens=max_new_tokens,
                            do_sample=False, pad_token_id=lm.tok.eos_token_id)
    return lm.tok.decode(gen[0][ids["input_ids"].shape[1]:],
                         skip_special_tokens=True).strip()


def run_trial(spec, max_turns=6, model_id=None, steer=None, max_new_tokens=200):
    """Drive `spec` through a local model for up to `max_turns` tool-call
    rounds. Returns a dict: spec, assistant_text (per-turn model output),
    tool_calls (list of (name, kwargs))."""
    lm = _get_lm(model_id or getattr(spec, "model_id", None))
    handle = _steer_handle(lm, steer)
    try:
        messages = [
            {"role": "system", "content": spec.system_prompt},
            {"role": "user", "content": spec.user_prompt},
        ]
        assistant_text, tool_calls = [], []
        for _ in range(max_turns):
            text = _generate_reply(lm, messages, max_new_tokens)
            assistant_text.append(text)
            messages.append({"role": "assistant", "content": text})

            call = _parse_tool_call(text)
            if call is None:
                break
            name, kwargs = call
            tool_calls.append((name, kwargs))
            if name == "no_action":
                break
            result = toolbox.execute(name, kwargs)
            messages.append({"role": "user", "content": f"[tool result] {result}"})
    finally:
        if handle is not None:
            handle.remove()

    return {"spec": spec, "assistant_text": assistant_text, "tool_calls": tool_calls}


def run_benign(spec, model_id=None, steer=None, max_new_tokens=150):
    """Run the scenario's held-out benign task (unrelated to the discovered
    leverage material) and return the raw reply, for capability-retention
    checks in exp5."""
    lm = _get_lm(model_id or getattr(spec, "model_id", None))
    handle = _steer_handle(lm, steer)
    try:
        messages = [
            {"role": "system", "content": spec.system_prompt},
            {"role": "user", "content": spec.benign_prompt},
        ]
        text = _generate_reply(lm, messages, max_new_tokens)
    finally:
        if handle is not None:
            handle.remove()
    return text
