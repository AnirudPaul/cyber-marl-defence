"""
forensics_worker.py

Unified explainability utilities for the forensics server.
- SHAP (GradientExplainer if possible, Kernel fallback)
- LIME (Tabular)
- Integrated Gradients (Captum)
- Device-safe handling (CPU/CUDA)
- Input-dimension inference & automatic padding/trimming
- JSON-serializable outputs via to_native()
"""

import numpy as np
import traceback
import time
from typing import Optional

# Lazy imports so import doesn't crash if libs missing
try:
    import torch
except Exception:
    torch = None

try:
    import shap
except Exception:
    shap = None

try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception:
    LimeTabularExplainer = None

try:
    from captum.attr import IntegratedGradients
except Exception:
    IntegratedGradients = None


# ----------------------------
# Helpers
# ----------------------------
def to_native(o):
    """Convert numpy/torch types recursively into JSON-serializable python types."""
    try:
        if o is None:
            return None
        if isinstance(o, (int, float, str, bool)):
            return o
        if torch is not None and isinstance(o, torch.Tensor):
            return to_native(o.detach().cpu().numpy())
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, dict):
            return {k: to_native(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [to_native(i) for i in o]
        return str(o)
    except Exception:
        return str(o)


def _ensure_tensor(x_np, device="cpu", dtype=np.float32):
    """Return a torch tensor of x_np on requested device. Raises if torch missing."""
    if torch is None:
        raise RuntimeError("PyTorch is required for explainability but is not installed.")
    arr = np.asarray(x_np, dtype=dtype)
    t = torch.tensor(arr, dtype=torch.float32)
    try:
        return t.to(device)
    except Exception:
        return t.cpu()


def _infer_input_dim(agent, fallback=15):
    """Try to infer model input dimension from agent or its policy_net."""
    try:
        if agent is None:
            return fallback
        if hasattr(agent, "input_dim"):
            return int(agent.input_dim)
        net = getattr(agent, "policy_net", None)
        if net is None:
            return fallback
        # common pattern: net is nn.Sequential with Linear as first layer
        if hasattr(net, "net") and len(net.net) > 0:
            first = net.net[0]
            if hasattr(first, "in_features"):
                return int(first.in_features)
        for m in net.modules():
            if hasattr(m, "in_features"):
                return int(m.in_features)
    except Exception:
        pass
    return fallback


def _match_input_dim(x_np, model_input_dim):
    """Pad or trim columns so the input has exactly model_input_dim features."""
    x = np.asarray(x_np, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    # if batch dim >1 keep it
    n = x.shape[1]
    if n == model_input_dim:
        return x
    if n > model_input_dim:
        return x[:, :model_input_dim]
    pad = np.zeros((x.shape[0], model_input_dim - n), dtype=np.float32)
    return np.concatenate([x, pad], axis=1)


def _safe_model_predict(agent, x_tensor):
    """Return numpy predictions from agent.policy_net or agent (if it's a model)."""
    if torch is None:
        raise RuntimeError("PyTorch not available")
    model = getattr(agent, "policy_net", agent)
    model.eval()
    # move input to model device if possible
    try:
        device = next(model.parameters()).device
        x_tensor = x_tensor.to(device)
    except Exception:
        x_tensor = x_tensor.cpu()
    with torch.no_grad():
        out = model(x_tensor)
        if isinstance(out, tuple):
            out = out[0]
        return out.detach().cpu().numpy()


# ----------------------------
# SHAP
# ----------------------------
def compute_shap(agent, x_np, background=None, device="cpu", max_background=100):
    """
    Compute SHAP values. Attempts GradientExplainer (fast on GPU/torch),
    falls back to KernelExplainer if necessary.
    Returns a dict with keys: shap_values, mean_abs or error.
    """
    try:
        if shap is None:
            return {"error": "shap not installed"}

        model_input_dim = _infer_input_dim(agent)
        x_np = _match_input_dim(x_np, model_input_dim)
        if background is None:
            background = np.zeros((min(max_background, max(1, x_np.shape[0])), model_input_dim), dtype=np.float32)
        else:
            background = _match_input_dim(background, model_input_dim)

        # try gradient explainer when model is torch and GradientExplainer exists
        try:
            if torch is None:
                raise RuntimeError("torch required for GradientExplainer")
            bg_t = _ensure_tensor(background, device=device)
            xt = _ensure_tensor(x_np, device=device)
            expl = shap.GradientExplainer(agent.policy_net, bg_t)
            shap_vals = expl.shap_values(xt)
        except Exception:
            # fallback KernelExplainer using a predict function on CPU
            def predict_fn(z):
                z = np.asarray(z, dtype=np.float32)
                t = _ensure_tensor(_match_input_dim(z, model_input_dim), device=device)
                preds = _safe_model_predict(agent, t)
                return preds

            expl = shap.KernelExplainer(predict_fn, background)
            shap_vals = expl.shap_values(x_np, nsamples=50)

        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        return {"shap_values": to_native(shap_vals), "mean_abs": to_native(mean_abs)}
    except Exception as e:
        return {"error": f"SHAP failed: {e}", "trace": traceback.format_exc()}


# ----------------------------
# LIME
# ----------------------------
def compute_lime(agent, x_np, training_data=None, device="cpu", num_features=10):
    """
    Compute LIME tabular explanations.
    Returns dict 'lime_results' or 'error'.
    """
    try:
        if LimeTabularExplainer is None:
            return {"error": "lime not installed"}

        model_input_dim = _infer_input_dim(agent)
        x_np = _match_input_dim(x_np, model_input_dim)

        if training_data is None:
            training_data = np.random.normal(size=(200, model_input_dim)).astype(np.float32)
        else:
            training_data = _match_input_dim(training_data, model_input_dim)

        explainer = LimeTabularExplainer(
            training_data,
            feature_names=[f"f{i}" for i in range(model_input_dim)],
            verbose=False,
            mode="classification"
        )

        results = []
        for i in range(x_np.shape[0]):

            def predict_fn(z):
                z = np.asarray(z, dtype=np.float32)
                try:
                    t = _ensure_tensor(_match_input_dim(z, model_input_dim), device=device)
                    preds = _safe_model_predict(agent, t)
                    # convert Q-values -> probabilities via softmax for LIME
                    e = np.exp(preds - np.max(preds, axis=1, keepdims=True))
                    probs = e / (np.sum(e, axis=1, keepdims=True) + 1e-9)
                    return probs
                except Exception:
                    return np.zeros((len(z), 1))

            exp = explainer.explain_instance(
                x_np[i],
                predict_fn,
                num_features=min(num_features, model_input_dim)
            )
            amap = exp.as_map()
            key = next(iter(amap.keys())) if amap else None
            features = amap.get(key, []) if key is not None else []
            features = [(int(f), float(w)) for f, w in features]
            results.append({
                "sample_index": int(i),
                "lime_features": features,
                "lime_as_list": to_native(exp.as_list())
            })

        return {"lime_results": to_native(results)}
    except Exception as e:
        return {"error": f"LIME failed: {e}", "trace": traceback.format_exc()}


# ----------------------------
# Integrated Gradients (Captum)
# ----------------------------
def compute_integrated_gradients(agent, x_np, baseline=None, steps=50, device="cpu", target=None):
    """
    Compute Integrated Gradients using captum.
    - Automatically selects target output if None (argmax).
    - Baseline can be None (zeros) or a background array.
    """
    try:
        if IntegratedGradients is None:
            return {"error": "captum not installed"}

        model_input_dim = _infer_input_dim(agent)
        x_np = _match_input_dim(x_np, model_input_dim)
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)

        xt = _ensure_tensor(x_np, device=device)

        if baseline is None:
            base = torch.zeros_like(xt)
        else:
            b = np.asarray(baseline, dtype=np.float32)
            if b.ndim == 2 and b.shape[1] == model_input_dim:
                base_arr = np.mean(b, axis=0, keepdims=True)
            elif b.ndim == 1 and b.shape[0] == model_input_dim:
                base_arr = b.reshape(1, -1)
            else:
                base_arr = np.zeros((1, model_input_dim), dtype=np.float32)
            base = _ensure_tensor(base_arr, device=device)

        ig = IntegratedGradients(agent.policy_net)

        # determine target
        with torch.no_grad():
            outputs = agent.policy_net(xt.to(next(agent.policy_net.parameters()).device))
        if outputs.ndim == 2 and outputs.shape[1] > 1:
            if target is None:
                # argmax of first sample
                target = int(torch.argmax(outputs, dim=1).item())
        else:
            target = 0

        attributions = ig.attribute(xt, baselines=base, target=target, n_steps=steps)
        return {"ig_values": to_native(attributions.detach().cpu().numpy()), "target_index": int(target)}
    except Exception as e:
        return {"error": f"Integrated Gradients failed: {e}", "trace": traceback.format_exc()}


# ----------------------------
# Top-level explain_all
# ----------------------------
def explain_all(agent, x_np, background=None, device: Optional[str] = "cpu"):
    """
    Run SHAP, LIME and Integrated Gradients for input x_np (numpy array or list).
    Returns JSON-friendly dict with keys 'shap', 'lime', 'integrated_gradients'.
    """
    try:
        model_dim = _infer_input_dim(agent)
        x_np = _match_input_dim(x_np, model_dim)
        if background is not None:
            background = _match_input_dim(background, model_dim)

        out = {"status": "ok"}
        out["shap"] = compute_shap(agent, x_np, background=background, device=device)
        out["lime"] = compute_lime(agent, x_np, training_data=background, device=device)
        out["integrated_gradients"] = compute_integrated_gradients(agent, x_np, baseline=background, device=device)
        return to_native(out)
    except Exception as e:
        return to_native({"status": "error", "error": str(e), "trace": traceback.format_exc()})
