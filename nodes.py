"""
The MIT License (MIT)

Copyright (c) 2015 Rolf Erik Lekang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# coding: utf-8
import os
import math
import logging
from typing import Optional, Literal

import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import trange

import comfy.samplers
import comfy.sample
import comfy.utils
import comfy.model_management
import comfy.ldm.modules.diffusionmodules.util
import comfy.ops
import folder_paths

# --- Optional deps ---
try:
    from safetensors import safe_open
except Exception:
    safe_open = None

try:
    import gguf
except Exception:
    gguf = None

log = logging.getLogger("ComfyUI-TwinFlow")


# ==============================================================================
# 0. Minimal Qwen component fallbacks (only used if model doesn't provide them)
# ==============================================================================

class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: float = 1.0):
        super().__init__()
        self.num_channels = int(num_channels)
        self.flip_sin_to_cos = bool(flip_sin_to_cos)
        self.downscale_freq_shift = float(downscale_freq_shift)
        self.scale = float(scale)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # Compute in fp32 for stability/speed; cast later.
        t = timesteps.float() * self.scale
        half_dim = self.num_channels // 2
        exponent = -math.log(10000.0) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - self.downscale_freq_shift)
        emb = torch.exp(exponent)
        emb = t[:, None] * emb[None, :]
        if self.flip_sin_to_cos:
            emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        else:
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int, act_fn: str = "silu", out_dim: Optional[int] = None, operations=None):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.act = nn.SiLU() if act_fn == "silu" else nn.ReLU()
        time_embed_dim_out = out_dim if out_dim is not None else time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, bias=True)

    def forward(self, sample: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        if condition is not None:
            sample = sample + condition
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


# ==============================================================================
# 1. Helper Functions
# ==============================================================================

def get_twinflow_patch_files():
    """List both 'diffusion_models' registry entries and any .gguf in those folders."""
    try:
        files = folder_paths.get_filename_list("diffusion_models")
    except Exception:
        files = []

    paths = folder_paths.get_folder_paths("diffusion_models")
    gguf_files = []
    for base in paths:
        if not os.path.exists(base):
            continue
        for root, _, filenames in os.walk(base, followlinks=True):
            for fn in filenames:
                if fn.lower().endswith(".gguf"):
                    full = os.path.join(root, fn)
                    rel = os.path.relpath(full, base)
                    gguf_files.append(rel)

    return sorted(list(set(files + gguf_files)))


def _load_twinflow_keys_safetensors(path: str, prefixes):
    if safe_open is None:
        return {}
    weights = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        needed = [k for k in keys if any(k.startswith(p) for p in prefixes)]
        for k in needed:
            weights[k] = f.get_tensor(k)
    return weights

_QK8_0 = 32
_BLOCK_Q8_0_SIZE = 2 + _QK8_0  # fp16(2 bytes) + 32*int8

_DT_Q8_0 = np.dtype([
    ("d",  "<f2"),          # little-endian float16
    ("qs", "i1", (_QK8_0,)), # int8[32]
], align=False)

def _dequant_q8_0_bytes_to_np(qbytes_u8: np.ndarray, n_elem: int, out_dtype=np.float16) -> np.ndarray:
    if n_elem % _QK8_0 != 0:
        raise ValueError(f"Q8_0 n_elem must be multiple of {_QK8_0}, got {n_elem}")
    nb = n_elem // _QK8_0
    expected = nb * _BLOCK_Q8_0_SIZE
    if int(qbytes_u8.nbytes) != expected:
        raise ValueError(f"Q8_0 bytes mismatch: expect {expected}, got {qbytes_u8.nbytes}")

    blocks = np.frombuffer(qbytes_u8, dtype=_DT_Q8_0, count=nb)
    d  = blocks["d"].astype(np.float32)          # (nb,)
    qs = blocks["qs"].astype(np.float32)         # (nb,32)
    out = (qs * d[:, None]).reshape(n_elem)      # float32

    return out.astype(out_dtype, copy=False)

def _load_twinflow_keys_gguf(path: str, prefixes):
    if gguf is None:
        raise ImportError("Loading GGUF models requires the 'gguf' package.")
    weights = {}
    reader = gguf.GGUFReader(path)

    F32 = getattr(gguf.GGMLQuantizationType, "F32", None)
    F16 = getattr(gguf.GGMLQuantizationType, "F16", None)
    BF16 = getattr(gguf.GGMLQuantizationType, "BF16", None)
    Q8_0 = getattr(gguf.GGMLQuantizationType, "Q8_0", None)

    for t in reader.tensors:
        name = t.name
        if not any(name.startswith(p) for p in prefixes):
            continue

        qtype = getattr(t, "tensor_type", None)
        if qtype is None:
            qtype = getattr(t, "type", None)
        
        shape = tuple(getattr(t, "shape", ()))
        if not shape:            
            shape = tuple(getattr(t, "dims", ()))

        raw = np.array(t.data, copy=True).view(np.uint8).reshape(-1)

        if qtype == F32:
            arr = np.array(t.data, copy=True)
            weights[name] = torch.from_numpy(arr).to(torch.float32)
        elif qtype == F16:
            arr = np.array(t.data, copy=True)
            weights[name] = torch.from_numpy(arr).to(torch.float16)
        elif qtype == BF16:
            u16 = np.array(t.data, copy=True).view(np.uint16)
            weights[name] = torch.from_numpy(u16).view(torch.bfloat16)
        elif qtype == Q8_0:
            qbytes = np.array(t.data, copy=True).view(np.uint8).reshape(-1)
            
            dims = np.array(t.shape, dtype=np.int64).tolist()
            np_shape = tuple(reversed(dims))

            fp = _dequant_q8_0_bytes_to_np(qbytes, n_elem=int(t.n_elements), out_dtype=np.float16)
            fp = fp.reshape(np_shape)
            
            weights[name] = torch.from_numpy(fp).to(torch.float16)
        else:            
            continue

    return weights


def load_specific_keys(ckpt_path: str, prefixes):
    """Load only TwinFlow patch keys from a safetensors/gguf/pt checkpoint."""
    ext = os.path.splitext(ckpt_path)[1].lower()
    try:
        if ext == ".safetensors":
            return _load_twinflow_keys_safetensors(ckpt_path, prefixes)

        if ext == ".gguf":
            return _load_twinflow_keys_gguf(ckpt_path, prefixes)

        # Fallback: torch checkpoint
        sd = comfy.utils.load_torch_file(ckpt_path)
        weights = {}
        for k in list(sd.keys()):
            if any(k.startswith(p) for p in prefixes):
                weights[k] = sd[k]
            # aggressively free
            del sd[k]
        return weights
    except Exception as e:
        log.warning("TwinFlow: failed to load patch weights from %s: %s", ckpt_path, e)
        return {}


def _safe_dtype_like(dtype: torch.dtype) -> torch.dtype:
    # fp8 params in some UNets; TwinFlow MLP weights are usually fp16/fp32.
    if dtype in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)):
        return torch.float16
    return dtype


def _infer_mlp_dims(sd: dict) -> Optional[tuple[int, int, int]]:
    """Infer (in_dim, hidden_dim, out_dim) from keys 'mlp.0.weight' and 'mlp.2.weight'."""
    w0 = sd.get("mlp.0.weight", None)
    w2 = sd.get("mlp.2.weight", None)
    if not (torch.is_tensor(w0) and torch.is_tensor(w2)):
        return None
    hidden_dim = int(w0.shape[0])
    in_dim = int(w0.shape[1])
    out_dim = int(w2.shape[0])
    return in_dim, hidden_dim, out_dim


# ==============================================================================
# 2. Sampler Logic (float32-first)
# ==============================================================================

class LinearTransport:
    # Matches TwinFlow's linear transport
    def alpha_in(self, t): return t
    def gamma_in(self, t): return 1 - t
    def alpha_to(self, t): return 1
    def gamma_to(self, t): return -1


class UnifiedSamplerImpl:
    """
    A ComfyUI-friendly implementation of TwinFlow unified sampling.
    - Model is assumed to return x0 (Comfy's standard).
    - We pass target_timestep as "sigma" into transformer_options; patcher maps it to model time-domain.
    """

    def __init__(self, model, sampling_order: int = 1, stochast_ratio: float = 0.0, extrapol_ratio: float = 0.0,
                 sampling_style: Literal["few", "any", "mul"] = "few"):
        self.model = model
        self.sampling_order = int(sampling_order)
        self.stochast_ratio = float(stochast_ratio)
        self.extrapol_ratio = float(extrapol_ratio)
        self.sampling_style = sampling_style
        self.transport = LinearTransport()
        self.invert_output = False  # keep for compatibility

    def forward_model(self, x: torch.Tensor, sigma: torch.Tensor, tt_sigma: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        # Copy model_options so we don't mutate upstream dicts.
        model_options = (kwargs.get("model_options", {}) or {}).copy()
        topts = (model_options.get("transformer_options", {}) or {}).copy()
        if tt_sigma is not None:
            # NOTE: tt_sigma is in "sigma domain"; patcher maps it to timestep domain.
            topts["target_timestep"] = tt_sigma
        model_options["transformer_options"] = topts

        denoise_mask = kwargs.get("denoise_mask", None)
        seed = kwargs.get("seed", None)

        x0 = self.model(x, sigma, denoise_mask=denoise_mask, model_options=model_options, seed=seed)

        # v = eps = (x - x0) / sigma  (sigma broadcast)
        sigma_clamped = sigma.clamp_min(1e-5)
        if sigma_clamped.ndim < x.ndim:
            sigma_clamped = sigma_clamped.view(-1, *([1] * (x.ndim - 1)))
        v = (x - x0) / sigma_clamped
        return -v if self.invert_output else v

    def reconstruction(self, x_t: torch.Tensor, F_t: torch.Tensor, t: torch.Tensor):
        """
        TwinFlow forward() does:
          dent = -1 (for linear transport in their code-path)
          z_hat = (x_t*gamma_to(t) - F_t*gamma_in(t))/dent
          x_hat = (F_t*alpha_in(t) - x_t*alpha_to(t))/dent
        """
        t_ = t.view(-1, *([1] * (x_t.ndim - 1)))
        dent = -1.0
        z_hat = (x_t * self.transport.gamma_to(t_) - F_t * self.transport.gamma_in(t_)) / dent
        x_hat = (F_t * self.transport.alpha_in(t_) - x_t * self.transport.alpha_to(t_)) / dent
        return x_hat, z_hat

    @torch.inference_mode()
    def sample(self, x: torch.Tensor, sigmas: torch.Tensor, extra_args=None, callback=None, disable=None):
        extra_args = {} if extra_args is None else extra_args
        x_cur = x

        # Extrapolation buffer (buffer_freq fixed to 1)
        prev_x_hat = None
        prev_z_hat = None

        for i in trange(len(sigmas) - 1, disable=disable):
            sigma_cur = sigmas[i]
            sigma_next = sigmas[i + 1]

            sigma_cur_t = sigma_cur.expand(x_cur.shape[0]).to(device=x_cur.device, dtype=torch.float32)
            sigma_next_t = sigma_next.expand(x_cur.shape[0]).to(device=x_cur.device, dtype=torch.float32)

            # Choose target sigma for "tt"
            if self.sampling_style == "few":
                sigma_tgt_t = torch.zeros_like(sigma_cur_t)
            elif self.sampling_style == "mul":
                sigma_tgt_t = sigma_cur_t
            else:  # "any"
                sigma_tgt_t = sigma_next_t

            v_cur = self.forward_model(x_cur, sigma_cur_t, tt_sigma=sigma_tgt_t, **extra_args)
            x_hat, z_hat = self.reconstruction(x_cur, v_cur, sigma_cur_t)

            # Extrapolation (one-step buffer)
            if self.extrapol_ratio > 0.0 and prev_x_hat is not None and prev_z_hat is not None:
                z_hat = z_hat + self.extrapol_ratio * (z_hat - prev_z_hat)
                x_hat = x_hat + self.extrapol_ratio * (x_hat - prev_x_hat)
            prev_x_hat, prev_z_hat = x_hat, z_hat

            # Stochastic mixing
            if self.stochast_ratio > 0.0:
                noi = torch.randn_like(x_cur)
                z_comp = z_hat * math.sqrt(1.0 - self.stochast_ratio) + noi * math.sqrt(self.stochast_ratio)
            else:
                z_comp = z_hat

            t_next_ = sigma_next_t.view(-1, *([1] * (x_cur.ndim - 1)))
            x_next = self.transport.gamma_in(t_next_) * x_hat + self.transport.alpha_in(t_next_) * z_comp

            # Second order correction (Heun-like trapezoid on v)
            if self.sampling_order == 2 and i < (len(sigmas) - 2):
                if self.sampling_style == "few":
                    sigma_tgt_next = torch.zeros_like(sigma_next_t)
                elif self.sampling_style == "mul":
                    sigma_tgt_next = sigma_next_t
                else:  # "any"
                    sigma_tgt_next = sigma_next_t

                v_next = self.forward_model(x_next, sigma_next_t, tt_sigma=sigma_tgt_next, **extra_args)
                dt = (sigma_next_t - sigma_cur_t).view(-1, *([1] * (x_cur.ndim - 1)))
                x_next = x_cur + 0.5 * dt * (v_cur + v_next)

            x_cur = x_next

            if callback is not None:
                callback({"x": x_cur, "i": i, "sigma": sigma_cur, "denoised": x_hat})

        return x_cur


# ==============================================================================
# 3. Architecture Patchers / Proxies
# ==============================================================================

class _TwinFlowTimestepEmbedder2(nn.Module):
    """
    Minimal t_embedder_2: timestep_embedding(., freq_size)->MLP
    Built dynamically from checkpoint shapes.
    """
    def __init__(self, in_dim: int = 256, hidden_dim: int = 1024, out_dim: int = 256, freq_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = int(freq_size)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )

    def forward(self, t: torch.Tensor, dtype: torch.dtype):
        # Compute sin/cos embedding in fp32, then cast to weight dtype.
        weight_dtype = self.mlp[0].weight.dtype
        t_freq = comfy.ldm.modules.diffusionmodules.util.timestep_embedding(t.float(), self.frequency_embedding_size).to(weight_dtype)
        out = self.mlp(t_freq)
        return out.to(dtype)


class TwinFlowTimestepEmbedderProxy(nn.Module):
    """
    Z-Image/Lumina-like:
      temb = original(t) + t_embedder_2( (target - t)*time_scale ) * abs(delta_t)
    delta_t is computed in "normalized time domain".
    We infer at runtime if `t` is already scaled (0..1000) or normalized (0..1).
    """
    def __init__(self, original_embedder: nn.Module, twinflow_embedder_2: nn.Module, time_scale: float):
        super().__init__()
        self.original = original_embedder
        self.twinflow_2 = twinflow_embedder_2
        self.time_scale = float(time_scale) if time_scale is not None else 1000.0
        self.target_timestep: Optional[torch.Tensor] = None

    def _ensure_device_dtype(self, ref: torch.Tensor, target_dtype: torch.dtype):
        # Only move when needed; avoid per-step overhead.
        p = next(self.twinflow_2.parameters())
        if p.device != ref.device:
            self.twinflow_2.to(ref.device)
        if p.dtype != target_dtype:
            self.twinflow_2.to(dtype=target_dtype)

    def forward(self, t: torch.Tensor, dtype: Optional[torch.dtype] = None):
        if dtype is None:
            try:
                dtype = next(self.original.parameters()).dtype
            except Exception:
                dtype = t.dtype

        t_emb = self.original(t, dtype)

        if self.target_timestep is None:
            return t_emb

        target_t = self.target_timestep.to(device=t.device, dtype=t.dtype)

        # scaled-domain heuristic
        t_abs_max = float(t.detach().abs().max().item()) if t.numel() else 0.0
        scaled_domain = (t_abs_max > 2.0) and (self.time_scale > 2.0)

        if scaled_domain:
            t_norm = t / self.time_scale
            target_norm = target_t / self.time_scale
        else:
            t_norm = t
            target_norm = target_t

        delta_abs = (t_norm - target_norm).abs()
        diff_in = (target_norm - t_norm) * self.time_scale

        self._ensure_device_dtype(ref=t_emb, target_dtype=t_emb.dtype)
        t_emb_2 = self.twinflow_2(diff_in, dtype=t_emb.dtype)

        t_emb = t_emb + t_emb_2 * delta_abs.unsqueeze(1).to(t_emb.dtype)
        return t_emb


# ---- Qwen embedder (fallback) ----

class TwinFlowQwenTimeEmbedder(nn.Module):
    """
    Minimal compatible 'time_text_embed_2' for Qwen-style models.
    Timesteps(scale=1000) assumes Comfy's Qwen uses normalized timesteps.
    """
    def __init__(self, embedding_dim: int):
        super().__init__()
        ops = comfy.ops.disable_weight_init
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0.0, scale=1000.0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=int(embedding_dim), operations=ops)

    def forward(self, timestep: torch.Tensor, hidden_states: torch.Tensor):
        tproj = self.time_proj(timestep)
        temb = self.timestep_embedder(tproj.to(dtype=hidden_states.dtype))
        return temb


class TwinFlowQwenEmbedderProxy(nn.Module):
    """
    Qwen-like:
      temb = original(t) + temb2(target_t) * ((t - target_t)/S)
    注意这里不取 abs，保留符号，和 TwinFlow/Qwen 参考实现一致。
    """
    def __init__(self, original_embedder: nn.Module, twinflow_embedder_2: nn.Module, scale_base: float = 1000.0):
        super().__init__()
        self.original = original_embedder
        self.twinflow_2 = twinflow_embedder_2
        self.scale_base = float(scale_base)
        self.target_timestep: Optional[torch.Tensor] = None

    def _ensure_device_dtype(self, ref: torch.Tensor, target_dtype: torch.dtype):
        p = next(self.twinflow_2.parameters())
        if p.device != ref.device:
            self.twinflow_2.to(ref.device)
        if p.dtype != target_dtype:
            self.twinflow_2.to(dtype=target_dtype)

    def forward(self, timestep: torch.Tensor, hidden_states: torch.Tensor, addition_t_cond=None):
        temb = self.original(timestep, hidden_states, addition_t_cond)

        if self.target_timestep is None:
            return temb

        target_t = self.target_timestep.to(device=timestep.device, dtype=timestep.dtype)

        # Heuristic: if timestep is already "scaled" (~0..1000), divide by 1000; else by 1.
        t_abs_max = float(timestep.detach().abs().max().item()) if timestep.numel() else 0.0
        denom = self.scale_base if t_abs_max > 2.0 else 1.0

        self._ensure_device_dtype(ref=temb, target_dtype=hidden_states.dtype)
        temb2 = self.twinflow_2(target_t, hidden_states)

        scale = ((timestep - target_t) / denom).unsqueeze(1).to(temb.dtype)
        temb = temb + temb2 * scale
        return temb


# ==============================================================================
# 4. Patcher Node
# ==============================================================================

class TwinFlowPatcher:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "patch_file": (get_twinflow_patch_files(),),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "sampling/twinflow"

    def patch(self, model, patch_file):
        model_patcher = model.clone()
        diffusion_model = model_patcher.model.diffusion_model

        ckpt_path = folder_paths.get_full_path("diffusion_models", patch_file)
        if ckpt_path is None:
            raise FileNotFoundError(f"File not found: {patch_file}")

        is_qwen = hasattr(diffusion_model, "time_text_embed")
        # Define target attribute name to be swapped during execution
        target_attr = "time_text_embed" if is_qwen else "t_embedder"

        prefixes = ["time_text_embed_2.", "transformer.time_text_embed_2."] if is_qwen else ["t_embedder_2.", "extended_weights.t_embedder_2."]

        weights = load_specific_keys(ckpt_path, prefixes)

        # Strip prefixes
        clean = {}
        for k, v in weights.items():
            for p in prefixes:
                if k.startswith(p):
                    clean[k[len(p):]] = v
                    break

        if not clean:
            log.warning("TwinFlow: no compatible keys found in %s", ckpt_path)
            return (model_patcher,)

        # Capture the current (original) embedder.
        # If it's already a proxy (from nested patchers or dirty state), try to unwrap.
        current_embedder = getattr(diffusion_model, target_attr)
        if isinstance(current_embedder, (TwinFlowTimestepEmbedderProxy, TwinFlowQwenEmbedderProxy)):
            original_module = current_embedder.original
        else:
            original_module = current_embedder

        # Construct the secondary embedder (t2)
        if is_qwen:
            # Infer embedding_dim
            try:
                embedding_dim = int(original_module.timestep_embedder.linear_1.out_features)
            except Exception:
                w = clean.get("timestep_embedder.linear_1.weight", None)
                embedding_dim = int(w.shape[0]) if torch.is_tensor(w) else 1024

            t2 = TwinFlowQwenTimeEmbedder(embedding_dim=embedding_dim)

            try:
                p0 = next(original_module.parameters())
                target_device = p0.device
                target_dtype = _safe_dtype_like(p0.dtype)
            except Exception:
                target_device = torch.device("cpu")
                target_dtype = torch.float16

            state = {k: v.to(dtype=target_dtype, device="cpu") if torch.is_tensor(v) else v for k, v in clean.items()}
            t2.load_state_dict(state, strict=False)
            t2.to(device=target_device, dtype=target_dtype)

            # Create proxy but DO NOT assign to diffusion_model yet
            proxy = TwinFlowQwenEmbedderProxy(original_module, t2, scale_base=1000.0)

        else:
            # Standard DiT logic
            dims = _infer_mlp_dims(clean) or (256, 1024, 256)
            in_dim, hidden_dim, out_dim = dims
            t2 = _TwinFlowTimestepEmbedder2(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, freq_size=256)

            try:
                p0 = next(original_module.parameters())
                target_device = p0.device
                target_dtype = _safe_dtype_like(p0.dtype)
            except Exception:
                target_device = torch.device("cpu")
                target_dtype = torch.float16

            state = {k: v.to(dtype=target_dtype, device="cpu") if torch.is_tensor(v) else v for k, v in clean.items()}
            t2.load_state_dict(state, strict=False)
            t2.to(device=target_device, dtype=target_dtype)

            time_scale = float(getattr(diffusion_model, "time_scale", 1000.0))
            proxy = TwinFlowTimestepEmbedderProxy(original_module, t2, time_scale=time_scale)

        # Define the wrapper that swaps the embedder temporarily during execution
        def twinflow_wrapper(model_apply_func, kwargs):
            c = kwargs.get("c", {}) or {}
            topts = c.get("transformer_options", {}) or {}
            tt_sigma = topts.get("target_timestep", None)

            mapped_tt = None
            if tt_sigma is not None:
                base_model = getattr(model_apply_func, "__self__", None)
                ms = getattr(base_model, "model_sampling", None)
                if ms is not None and hasattr(ms, "timestep"):
                    try:
                        mapped_tt = ms.timestep(tt_sigma).float()
                    except Exception:
                        mapped_tt = tt_sigma
                else:
                    mapped_tt = tt_sigma

            # Set target on proxy
            proxy.target_timestep = mapped_tt

            # Swap embedder
            # We use getattr/setattr on the diffusion_model instance captured in closure
            # This is the shared instance, so we must restore it.
            
            # NOTE: diffusion_model here is the specific instance loaded in memory.
            previous_embedder = getattr(diffusion_model, target_attr)
            setattr(diffusion_model, target_attr, proxy)

            try:
                x = kwargs["input"]
                t = kwargs["timestep"]
                return model_apply_func(x, t, **c)
            finally:
                # Restore original embedder
                setattr(diffusion_model, target_attr, previous_embedder)
                proxy.target_timestep = None

        model_patcher.set_model_unet_function_wrapper(twinflow_wrapper)
        return (model_patcher,)


# ==============================================================================
# 5. Nodes (Sampler / Scheduler)
# ==============================================================================

class TwinFlowSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampling_method": (["euler", "heun"], {"default": "euler"}),
                "stochast_ratio": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "extrapol_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sampling_style": (["few", "any", "mul"], {"default": "few"}),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"
    FUNCTION = "get_sampler"

    def get_sampler(self, sampling_method, stochast_ratio, extrapol_ratio, sampling_style):
        sampling_order = 1 if sampling_method == "euler" else 2

        def sampler_function(model, x, sigmas, extra_args=None, callback=None, disable=None):
            tf = UnifiedSamplerImpl(model, sampling_order, stochast_ratio, extrapol_ratio, sampling_style=sampling_style)
            sigmas_f = sigmas.to(dtype=torch.float32, device=x.device)
            return tf.sample(x, sigmas_f, extra_args=extra_args, callback=callback, disable=disable)

        return (comfy.samplers.KSAMPLER(sampler_function),)


class TwinFlowScheduler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {"default": 2, "min": 1, "max": 100, "step": 1}),
                "dist_ctrl_a": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "dist_ctrl_b": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "dist_ctrl_c": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "gap_start": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 1.0, "step": 0.001}),
                "gap_end": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"
    FUNCTION = "get_sigmas"

    @staticmethod
    def kumaraswamy_transform(t: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
        return (1.0 - (1.0 - t.pow(a)).pow(b)).pow(c)

    def get_sigmas(self, steps, dist_ctrl_a, dist_ctrl_b, dist_ctrl_c, gap_start, gap_end):
        num_steps = int(steps)
        t = torch.linspace(0.0, 1.0, num_steps, dtype=torch.float32)
        t_scaled = gap_start + t * (1.0 - gap_end - gap_start)
        t_steps = self.kumaraswamy_transform(t_scaled, dist_ctrl_a, dist_ctrl_b, dist_ctrl_c)
        sigmas = 1.0 - t_steps
        sigmas = torch.cat([sigmas, torch.zeros(1, dtype=torch.float32)])
        return (sigmas,)


# ==============================================================================
# 6. Unified Node (KSampler)
# ==============================================================================

class TwinFlowKSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 100, "step": 1}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampling_method": (["euler", "heun"], {"default": "euler"}),
                "sampling_style": (["few", "any", "mul"], {"default": "few"}),
                "stochast_ratio": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "extrapol_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dist_ctrl_a": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "dist_ctrl_b": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "dist_ctrl_c": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "gap_start": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 1.0, "step": 0.001}),
                "gap_end": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.001}),                
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/twinflow"

    @staticmethod
    def kumaraswamy_transform(t: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
        return (1.0 - (1.0 - t.pow(a)).pow(b)).pow(c)

    def sample(
        self,
        model,
        seed,
        steps,
        cfg,
        sampling_method,
        sampling_style,
        stochast_ratio,
        extrapol_ratio,
        dist_ctrl_a,
        dist_ctrl_b,
        dist_ctrl_c,
        gap_start,
        gap_end,
        positive,
        negative,
        latent_image,
        denoise,
    ):
        # ---- 1) sigmas (float32) ----
        steps = int(steps)
        total_steps = steps
        if denoise < 1.0:
            if denoise <= 0.0:
                return (latent_image,)
            total_steps = int(round(steps / denoise))

        t = torch.linspace(0.0, 1.0, total_steps, dtype=torch.float32)
        t_scaled = gap_start + t * (1.0 - gap_end - gap_start)
        t_steps = self.kumaraswamy_transform(t_scaled, dist_ctrl_a, dist_ctrl_b, dist_ctrl_c)
        sigmas = 1.0 - t_steps
        sigmas = torch.cat([sigmas, torch.zeros(1, dtype=torch.float32)])
        sigmas = sigmas[-(steps + 1):].to(device=model.load_device, dtype=torch.float32)

        # ---- 2) sampler ----
        sampling_order = 1 if sampling_method == "euler" else 2

        def sampler_function(model_, x, sigmas_, extra_args=None, callback=None, disable=None):
            tf = UnifiedSamplerImpl(model_, sampling_order, stochast_ratio, extrapol_ratio, sampling_style=sampling_style)
            sigmas_f = sigmas_.to(dtype=torch.float32, device=x.device)
            return tf.sample(x, sigmas_f, extra_args=extra_args, callback=callback, disable=disable)

        ksampler = comfy.samplers.KSAMPLER(sampler_function)

        # ---- 3) run (mirror common_ksampler essentials) ----
        latent = latent_image.copy()
        latent_samples = comfy.sample.fix_empty_latent_channels(model, latent["samples"])
        latent["samples"] = latent_samples

        noise = comfy.sample.prepare_noise(latent_samples, seed)
        noise_mask = latent.get("noise_mask", None)

        samples = comfy.sample.sample_custom(
            model,
            noise,
            cfg,
            ksampler,
            sigmas,
            positive,
            negative,
            latent_samples,
            noise_mask=noise_mask,
            seed=seed,
        )

        out = latent.copy()
        out["samples"] = samples
        return (out,)


NODE_CLASS_MAPPINGS = {
    "TwinFlowSampler": TwinFlowSampler,
    "TwinFlowScheduler": TwinFlowScheduler,
    "TwinFlowPatcher": TwinFlowPatcher,
    "TwinFlowKSampler": TwinFlowKSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TwinFlowSampler": "TwinFlow Sampler",
    "TwinFlowScheduler": "TwinFlow Scheduler",
    "TwinFlowPatcher": "TwinFlow Model Patcher",
    "TwinFlowKSampler": "TwinFlow KSampler",
}