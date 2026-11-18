"""Pure-torch core for the INT8 LoRA file format.

This module deliberately imports nothing from ComfyUI so it can be tested
and reused on its own. See FORMAT.md for the on-disk layout.

Summary of the format (version 1):

* A normal safetensors file.
* Every large float matrix of the source LoRA/LoKr/LoHa is stored as an int8
  tensor under its *original key*, with one float32 scale per row (per column
  for tall matrices) stored under ``<key>.int8_scale``. Optionally the values
  are ConvRot-rotated first.
* Everything else (alpha, dora_scale, 1-D tensors, tiny factors) is stored
  unchanged.
* File metadata marks the file and records the scheme, so loaders can reject
  files they do not understand instead of silently producing garbage.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FORMAT_VERSION = "1"
SCHEME_ABSMAX = "absmax"
ROTATION_NONE = "none"
ROTATION_CONVROT = "convrot"
ROTATIONS = (ROTATION_NONE, ROTATION_CONVROT)  # what a file can contain
ROTATION_AUTO = "auto"  # a request option only: try both, keep the lower SSE

# Metadata keys with this prefix belong to this format; source-file keys that
# happen to use it are not carried over.
META_PREFIX = "int8_lora_"
META_FORMAT = "int8_lora_format"
META_SCHEME = "int8_lora_scheme"
META_ROTATION = "int8_lora_rotation"
META_DTYPE = "int8_lora_dtype"
META_CONVROT_GROUP = "int8_lora_convrot_group"  # largest group size, e.g. "256"

SCALE_SUFFIX = ".int8_scale"

# ConvRot: orthonormal Hadamard rotation applied in groups of 256/64/16/4
# values (powers of four, same construction as PotatoForge's ConvRot). For each
# tensor the largest group size that divides the rotated axis is used; if none
# divides it, the tensor is stored unrotated. Both choices depend only on the
# tensor shape, so the file needs no per-tensor flags.
CONVROT_GROUPS = (256, 64, 16, 4)
DEFAULT_CONVROT_GROUP = 256

# Tensors with fewer elements than this stay in their original dtype. Tiny
# factors (for example a 16x16 LoKr w1) save almost nothing and are the most
# sensitive to quantization error.
MIN_QUANT_NUMEL = 1024

# Keys that are never quantized, whatever their shape.
SKIP_SUFFIXES = (".alpha", ".dora_scale", ".reshape_weight", ".diff_b")

_DTYPE_NAMES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
_DTYPE_BY_TORCH = {v: k for k, v in _DTYPE_NAMES.items()}


# --------------------------------------------------------------------------
# INT8 primitives
# --------------------------------------------------------------------------

def _scale_axis(shape: tuple[int, ...]) -> int:
    """Axis that gets one scale per index.

    Rows (axis 0) by default. For tall 2-D matrices (more rows than columns,
    e.g. ``lora_up`` of shape ``(out, rank)``) columns are used instead, so the
    number of scales is the *smaller* dimension and the scale overhead stays
    negligible. Square matrices use rows, which keeps loading unambiguous.
    """
    if len(shape) == 2 and shape[0] > shape[1]:
        return 1
    return 0


@lru_cache(maxsize=8)
def _hadamard(group: int) -> torch.Tensor:
    """Normalized power-of-four Hadamard matrix. Symmetric and orthonormal, so
    it is its own inverse."""
    base = torch.tensor(
        [
            [1.0, 1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    matrix = base
    size = 4
    while size < group:
        matrix = torch.kron(matrix, base)
        size *= 4
    return matrix / (group**0.5)


def _convrot_group_for(length: int, max_group: int) -> int:
    """Group size used for an axis of ``length`` values, or 0 for no rotation."""
    for group in CONVROT_GROUPS:
        if group <= max_group and length % group == 0:
            return group
    return 0


def _rotate_each_row(x: torch.Tensor, max_group: int) -> torch.Tensor:
    """Rotate every row of a 2-D tensor, one block of ``group`` values at a time.

    "Rotate" means multiplying a block by the Hadamard matrix. The block size is
    chosen from the row length; if no block size fits, ``x`` is returned
    unchanged. Self-inverse, so calling it again undoes it.
    """
    group = _convrot_group_for(x.shape[-1], max_group)
    if group == 0:
        return x
    h = _hadamard(group).to(device=x.device)
    rotated = x.reshape(x.shape[0], x.shape[1] // group, group) @ h
    return rotated.reshape(x.shape)


def _rotate(work: torch.Tensor, axis: int, max_group: int) -> torch.Tensor:
    """Rotate the numbers that share one scale.

    ``axis`` is the scale axis from :func:`_scale_axis`: 0 means one scale per
    row (the numbers sharing a scale lie along the row, so each row is rotated;
    conv tensors are flattened first), 1 means one scale per column (they lie
    down the column, so each column is rotated). Self-inverse, so the same call
    undoes it.
    """
    if axis == 1:
        return _rotate_each_row(work.t().contiguous(), max_group).t().contiguous()
    flat = work.reshape(work.shape[0], -1)
    return _rotate_each_row(flat, max_group).reshape(work.shape)


def is_rotated(shape: tuple[int, ...], rotation: str, max_group: int) -> bool:
    """Whether a tensor of this shape is rotated under the given setting."""
    if rotation != ROTATION_CONVROT:
        return False
    if _scale_axis(shape) == 1:
        length = shape[0]
    else:
        length = 1
        for dim in shape[1:]:
            length *= dim
    return _convrot_group_for(length, max_group) != 0


def quantize_absmax(
    tensor: torch.Tensor,
    rotation: str = ROTATION_NONE,
    max_group: int = DEFAULT_CONVROT_GROUP,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a float tensor (2-D or more) to int8 with absmax scales.

    Inputs are validated by :func:`quantize_state_dict`, not here.

    Returns ``(q, scale)`` where ``scale`` is float32 and has one entry per
    row (or per column for tall 2-D matrices, see :func:`_scale_axis`).
    Tensors with more than two dims (conv LoRAs) are scaled per output
    channel (dim 0); ``q`` always keeps the original shape. With
    ``rotation="convrot"`` the values are Hadamard-rotated before scaling.
    """
    work = tensor.detach().to(torch.float32)
    if not bool(torch.isfinite(work).all()):
        raise ValueError("tensor contains NaN or Inf and cannot be quantized")
    axis = _scale_axis(tuple(work.shape))
    if rotation == ROTATION_CONVROT:
        work = _rotate(work, axis, max_group)
    tiny = torch.finfo(torch.float32).tiny
    if axis == 1:
        scale = (work.abs().amax(dim=0) / 127.0).clamp_min(tiny)
        q = torch.round(work / scale[None, :])
    else:
        flat = work.reshape(work.shape[0], -1)
        scale = (flat.abs().amax(dim=1) / 127.0).clamp_min(tiny)
        q = torch.round(flat / scale[:, None]).reshape(work.shape)
    return q.clamp_(-127, 127).to(torch.int8).contiguous(), scale.contiguous()


def dequantize_absmax(
    q: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    rotation: str = ROTATION_NONE,
    max_group: int = DEFAULT_CONVROT_GROUP,
) -> torch.Tensor:
    """Inverse of :func:`quantize_absmax`; the axis is inferred from ``scale``."""
    axis = _scale_axis(tuple(q.shape))
    if scale.shape[0] != q.shape[axis]:
        raise ValueError(
            f"scale has {scale.shape[0]} entries but axis {axis} of {tuple(q.shape)} "
            f"has {q.shape[axis]}"
        )
    shape = [1] * q.ndim
    shape[axis] = q.shape[axis]
    out = q.to(torch.float32) * scale.to(torch.float32).reshape(shape)
    if rotation == ROTATION_CONVROT:
        out = _rotate(out, axis, max_group)
    return out.to(dtype)


# --------------------------------------------------------------------------
# Quantizing a state dict
# --------------------------------------------------------------------------

@dataclass
class TensorStat:
    key: str
    sse: float
    energy: float  # sum of squares of the original tensor

    @property
    def relative_sse(self) -> float:
        return self.sse / self.energy if self.energy > 0 else 0.0


@dataclass
class QuantizeReport:
    stats: list[TensorStat] = field(default_factory=list)
    kept_float: int = 0
    bytes_before: int = 0
    bytes_after: int = 0
    dtype_name: str = "float32"
    rotation: str = ROTATION_NONE
    rotated_count: int = 0
    carried_metadata: int = 0  # source-file metadata keys copied into the output
    # Filled by rotation="auto": total SSE of each candidate setting.
    tried: dict[str, float] = field(default_factory=dict)

    @property
    def quantized_count(self) -> int:
        return len(self.stats)

    @property
    def total_sse(self) -> float:
        return sum(s.sse for s in self.stats)

    @property
    def total_energy(self) -> float:
        return sum(s.energy for s in self.stats)

    @property
    def relative_sse(self) -> float:
        energy = self.total_energy
        return self.total_sse / energy if energy > 0 else 0.0

    @property
    def worst(self) -> TensorStat | None:
        return max(self.stats, key=lambda s: s.relative_sse, default=None)


def _is_quantizable(key: str, tensor: torch.Tensor, min_numel: int) -> bool:
    return (
        tensor.is_floating_point()
        and tensor.ndim >= 2
        and tensor.numel() >= min_numel
        and not key.endswith(SKIP_SUFFIXES)
    )


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _quantize_pass(
    state_dict: dict[str, torch.Tensor],
    candidates: set[str],
    dtype: torch.dtype,
    rotation: str,
    max_group: int,
) -> tuple[dict[str, torch.Tensor], QuantizeReport]:
    """Quantize every candidate tensor with one fixed rotation setting."""
    report = QuantizeReport(dtype_name=_DTYPE_BY_TORCH[dtype], rotation=rotation)
    out: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        report.bytes_before += _tensor_bytes(tensor)
        if key not in candidates:
            kept = tensor.detach().contiguous().clone()
            out[key] = kept
            report.kept_float += 1
            report.bytes_after += _tensor_bytes(kept)
            continue

        q, scale = quantize_absmax(tensor, rotation, max_group)
        original = tensor.detach().to(torch.float32)
        restored = dequantize_absmax(q, scale, dtype, rotation, max_group).to(torch.float32)
        if is_rotated(tuple(tensor.shape), rotation, max_group):
            report.rotated_count += 1
        sse = float((original - restored).pow(2).sum(dtype=torch.float64))
        energy = float(original.pow(2).sum(dtype=torch.float64))
        report.stats.append(TensorStat(key, sse, energy))

        out[key] = q
        out[key + SCALE_SUFFIX] = scale
        report.bytes_after += _tensor_bytes(q) + _tensor_bytes(scale)
    return out, report


def quantize_state_dict(
    state_dict: dict[str, torch.Tensor],
    min_numel: int = MIN_QUANT_NUMEL,
    rotation: str = ROTATION_AUTO,
    max_group: int = DEFAULT_CONVROT_GROUP,
    source_metadata: dict[str, str] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, str], QuantizeReport]:
    """Quantize a LoRA state dict.

    ``rotation`` is ``"none"``, ``"convrot"`` or ``"auto"``. With ``"auto"`` the
    file is quantized both ways and the setting with the lower total SSE is
    kept (``"none"`` on a tie, or when ConvRot would rotate nothing).

    ``source_metadata`` (the header metadata of the original file, for example
    kohya ``ss_*`` keys or trigger words) is copied into the output metadata so
    nothing the source carried is lost.

    Returns ``(tensors, metadata, report)`` ready for :func:`save_int8_lora`.
    The SSE in the report compares each original tensor with exactly what the
    loader will reconstruct (including the inverse rotation and the cast back
    to the stored dtype).
    """
    if rotation != ROTATION_AUTO and rotation not in ROTATIONS:
        raise ValueError(f"unknown rotation '{rotation}'")
    if max_group not in CONVROT_GROUPS:
        raise ValueError(f"convrot group must be one of {CONVROT_GROUPS}")
    for key in state_dict:
        if key.endswith(SCALE_SUFFIX):
            raise ValueError(
                f"'{key}' looks like an INT8 LoRA scale tensor; "
                "this file is probably already quantized"
            )

    candidates = {k for k, t in state_dict.items() if _is_quantizable(k, t, min_numel)}
    if not candidates:
        raise ValueError("no tensors large enough to quantize were found")

    dtype_counts = Counter(state_dict[k].dtype for k in candidates)
    target_dtype = dtype_counts.most_common(1)[0][0]
    if target_dtype not in _DTYPE_BY_TORCH:
        target_dtype = torch.float32

    if rotation == ROTATION_AUTO:
        plain, plain_report = _quantize_pass(
            state_dict, candidates, target_dtype, ROTATION_NONE, max_group
        )
        rotated, rotated_report = _quantize_pass(
            state_dict, candidates, target_dtype, ROTATION_CONVROT, max_group
        )
        tried = {
            ROTATION_NONE: plain_report.total_sse,
            ROTATION_CONVROT: rotated_report.total_sse,
        }
        if rotated_report.rotated_count > 0 and rotated_report.total_sse < plain_report.total_sse:
            out, report = rotated, rotated_report
        else:
            out, report = plain, plain_report
        report.tried = tried
        del plain, rotated
    else:
        out, report = _quantize_pass(state_dict, candidates, target_dtype, rotation, max_group)

    carried = {
        str(k): str(v)
        for k, v in (source_metadata or {}).items()
        if not str(k).startswith(META_PREFIX)
    }
    report.carried_metadata = len(carried)
    metadata = {
        **carried,
        META_FORMAT: FORMAT_VERSION,
        META_SCHEME: SCHEME_ABSMAX,
        META_ROTATION: report.rotation,
        META_DTYPE: report.dtype_name,
    }
    if report.rotation == ROTATION_CONVROT:
        metadata[META_CONVROT_GROUP] = str(max_group)
    return out, metadata, report


def format_report(report: QuantizeReport, source: str = "", target: str = "") -> str:
    """Human-readable summary of a quantization run."""
    mib = 1024 * 1024
    lines = []
    if source or target:
        lines.append(f"{source} -> {target}")
    lines.append(
        f"size: {report.bytes_before / mib:.1f} MiB -> {report.bytes_after / mib:.1f} MiB "
        f"({100.0 * report.bytes_after / max(report.bytes_before, 1):.1f}% of original)"
    )
    lines.append(
        f"tensors: {report.quantized_count} quantized to int8, "
        f"{report.kept_float} kept as float (stored dtype {report.dtype_name})"
    )
    if report.rotation == ROTATION_CONVROT:
        lines.append(
            f"rotation: convrot ({report.rotated_count} of {report.quantized_count} "
            "quantized tensors rotated)"
        )
    else:
        lines.append("rotation: none")
    if report.tried:
        lines.append(
            f"auto: SSE with none {report.tried[ROTATION_NONE]:.6g} vs convrot "
            f"{report.tried[ROTATION_CONVROT]:.6g} -> kept {report.rotation}"
        )
    if report.carried_metadata:
        lines.append(f"metadata: {report.carried_metadata} source keys carried over")
    else:
        lines.append("metadata: none in the source file")
    lines.append(f"SSE (original vs int8): {report.total_sse:.6g}")
    lines.append(
        f"relative SSE: {report.relative_sse:.3e} "
        f"(SSE / sum of squares of the original = {100.0 * report.relative_sse:.4f}%)"
    )
    worst = report.worst
    if worst is not None:
        lines.append(
            f"worst tensor: {worst.key} (relative SSE {worst.relative_sse:.3e}, "
            f"SSE {worst.sse:.6g})"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Saving and loading
# --------------------------------------------------------------------------

def save_int8_lora(
    path: str, tensors: dict[str, torch.Tensor], metadata: dict[str, str]
) -> None:
    save_file(tensors, path, metadata=metadata)


def read_source_metadata(path: str) -> dict[str, str]:
    """Header metadata of a source LoRA, or ``{}`` if the file has none or is not
    a readable safetensors file (for example a ``.pt`` checkpoint)."""
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            return dict(f.metadata() or {})
    except Exception:
        return {}


def _validate_metadata(
    metadata: dict[str, str], path: str
) -> tuple[torch.dtype, str, int]:
    """Check file metadata; return ``(dtype, rotation, max_group)``."""
    if META_FORMAT not in metadata:
        raise ValueError(f"{path} is not an INT8 LoRA file (missing '{META_FORMAT}')")
    version = metadata.get(META_FORMAT)
    if version != FORMAT_VERSION:
        raise ValueError(
            f"{path}: unsupported INT8 LoRA format version '{version}' "
            f"(this build reads version '{FORMAT_VERSION}')"
        )
    scheme = metadata.get(META_SCHEME)
    if scheme != SCHEME_ABSMAX:
        raise ValueError(f"{path}: unsupported quantization scheme '{scheme}'")
    rotation = metadata.get(META_ROTATION, ROTATION_NONE)
    if rotation not in ROTATIONS:
        raise ValueError(
            f"{path}: rotation '{rotation}' is not supported by this build; "
            "update ComfyUI-INT8-LoRA"
        )
    max_group = DEFAULT_CONVROT_GROUP
    if rotation == ROTATION_CONVROT:
        try:
            max_group = int(metadata.get(META_CONVROT_GROUP, DEFAULT_CONVROT_GROUP))
        except ValueError:
            max_group = 0
        if max_group not in CONVROT_GROUPS:
            raise ValueError(f"{path}: invalid convrot group in metadata")
    dtype_name = metadata.get(META_DTYPE, "float32")
    if dtype_name not in _DTYPE_NAMES:
        raise ValueError(f"{path}: unsupported stored dtype '{dtype_name}'")
    return _DTYPE_NAMES[dtype_name], rotation, max_group


def load_int8_lora_with_metadata(
    path: str, device: str = "cpu"
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Load an INT8 LoRA file.

    Returns ``(state_dict, metadata)``: an ordinary float state dict, and the
    source file's original metadata (this format's own ``int8_lora_*`` keys are
    removed), which is what ComfyUI's stock LoRA loader would see.
    """
    out: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        header = dict(f.metadata() or {})
        dtype, rotation, max_group = _validate_metadata(header, path)
        keys = set(f.keys())
        scale_keys = {k for k in keys if k.endswith(SCALE_SUFFIX)}

        for scale_key in scale_keys:
            if scale_key[: -len(SCALE_SUFFIX)] not in keys:
                raise ValueError(f"{path}: scale '{scale_key}' has no matching tensor")

        for key in keys - scale_keys:
            tensor = f.get_tensor(key)
            scale_key = key + SCALE_SUFFIX
            if scale_key in scale_keys:
                if tensor.dtype != torch.int8:
                    raise ValueError(f"{path}: '{key}' has a scale but is {tensor.dtype}")
                q = tensor.to(device)
                scale = f.get_tensor(scale_key).to(device)
                out[key] = dequantize_absmax(q, scale, dtype, rotation, max_group)
            else:
                if tensor.dtype == torch.int8:
                    raise ValueError(f"{path}: int8 tensor '{key}' has no scale")
                out[key] = tensor.to(device)
    source = {k: v for k, v in header.items() if not k.startswith(META_PREFIX)}
    return out, source


def load_int8_lora(path: str, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Load an INT8 LoRA file and return an ordinary float state dict."""
    return load_int8_lora_with_metadata(path, device)[0]


def sanitize_output_name(name: str, fallback_stem: str) -> str:
    """Turn user text into a safe ``*.safetensors`` filename (no directories)."""
    name = (name or "").strip() or f"{fallback_stem}_int8"
    if name.lower().endswith(".safetensors"):
        name = name[: -len(".safetensors")]
    if not name or name in {".", ".."} or any(c in name for c in '/\\:*?"<>|'):
        raise ValueError(f"invalid output name: {name!r}")
    return name + ".safetensors"
