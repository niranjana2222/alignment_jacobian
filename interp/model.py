"""
Local open-weight model wrapper.

Gradients and hidden states are required by the lenses and Jacobian readouts,
which a served API cannot provide, so the interp module runs an open-weight
subject model locally instead of the behavioral harness's API-backed one.
`LocalLM` is a thin wrapper around a HF causal LM exposing exactly the surface
the rest of interp/ needs: per-layer hidden states, the decoder layer modules
(for steering hooks), and the unembed/final-norm pair (for reading a mid-layer
residual out to vocab logits).
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def _pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LocalLM:
    def __init__(self, model_id=None, device=None, dtype=None):
        self.model_id = model_id or DEFAULT_MODEL_ID
        self.device = device or _pick_device()
        self.dtype = dtype or (torch.float32 if self.device == "cpu"
                               else torch.float16)

        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=self.dtype, output_hidden_states=True)
        self.model.to(self.device)
        self.model.eval()

        base = getattr(self.model, "model", self.model)
        self.layers = base.layers
        self.n_layers = len(self.layers)
        self.d_model = self.model.config.hidden_size
        self._base = base

    def encode(self, text):
        return self.tok(text, return_tensors="pt").to(self.device)

    @torch.no_grad()
    def hidden_states(self, text):
        """Return (per-layer residual list [n_layers+1] of [seq, d_model], the
        raw model output). Index 0 is the embedding output; 1..n_layers are
        post-block residuals, matching what estimate_directions expects."""
        ids = self.encode(text)
        out = self.model(**ids, output_hidden_states=True)
        hs = [h[0] for h in out.hidden_states]
        return hs, out

    def final_norm(self, h):
        """Apply the model's final RMSNorm/LayerNorm to a residual vector so
        it can be unembedded like the true final-layer output."""
        norm = getattr(self._base, "norm", None)
        return norm(h) if norm is not None else h

    def unembed(self, h):
        return self.model.lm_head(h)
