"""ComfyUI nodes for INT8 LoRA files.

Only ComfyUI's regular loader functions are used (folder_paths,
comfy.utils.load_torch_file, comfy.sd.load_lora_for_models). Nothing is
monkeypatched. The one internal detail we touch is the
folder_paths.folder_names_and_paths dict, to give the INT8 folder an extension
filter (see below).
"""

from __future__ import annotations

import inspect
import logging
import os

import comfy.model_management
import comfy.sd
import comfy.utils
import folder_paths

from . import int8_lora_core as core

LOG = logging.getLogger("ComfyUI-INT8-LoRA")

INT8_FOLDER = "loras_int8"
NONE_CHOICE = "None"
NO_FOLDERS = "(no subfolders in loras)"
STACK_SLOTS = 3

# INT8 LoRAs live in their own folder (models/loras_int8) so the regular LoRA
# dropdowns never offer files that stock loaders cannot read.
_int8_dir = os.path.join(folder_paths.models_dir, INT8_FOLDER)
os.makedirs(_int8_dir, exist_ok=True)
folder_paths.add_model_folder_path(INT8_FOLDER, _int8_dir, is_default=True)
folder_paths.folder_names_and_paths[INT8_FOLDER] = (
    folder_paths.get_folder_paths(INT8_FOLDER),
    {".safetensors"},
)


# Newer ComfyUI attaches the LoRA file's metadata to the patched model; older
# versions do not accept the argument.
_ACCEPTS_LORA_METADATA = (
    "lora_metadata" in inspect.signature(comfy.sd.load_lora_for_models).parameters
)


def _int8_choices() -> list[str]:
    return [NONE_CHOICE] + folder_paths.get_filename_list(INT8_FOLDER)


def _int8_target(relative: str) -> str:
    """Absolute output path for a '/'-separated path under the INT8 folder."""
    base = os.path.abspath(folder_paths.get_folder_paths(INT8_FOLDER)[0])
    target = os.path.abspath(os.path.join(base, *relative.split("/")))
    if os.path.commonpath([base, target]) != base:
        raise ValueError("output path escapes the INT8 folder")
    return target


def _quantize_to_file(source: str, target: str, rotation: str) -> core.QuantizeReport:
    """Quantize one LoRA file to ``target`` (never touching the source)."""
    state_dict = comfy.utils.load_torch_file(source, safe_load=True)
    tensors, metadata, report = core.quantize_state_dict(
        state_dict,
        rotation=rotation,
        source_metadata=core.read_source_metadata(source),
    )
    del state_dict

    # Write to a temp name first so a failure never leaves a truncated file.
    os.makedirs(os.path.dirname(target), exist_ok=True)
    temp = target + ".tmp"
    try:
        core.save_int8_lora(temp, tensors, metadata)
        os.replace(temp, target)
    finally:
        if os.path.exists(temp):
            os.remove(temp)
    return report


def _report_result(text: str) -> dict:
    """Log a report and return it both to the report output and to the UI."""
    LOG.info("\n%s", text)
    return {"ui": {"text": [text]}, "result": (text,)}


def _apply_int8_lora(model, clip, lora_name, strength_model, strength_clip):
    path = folder_paths.get_full_path_or_raise(INT8_FOLDER, lora_name)
    state_dict, metadata = core.load_int8_lora_with_metadata(path)
    extra = {"lora_metadata": metadata or None} if _ACCEPTS_LORA_METADATA else {}
    return comfy.sd.load_lora_for_models(
        model, clip, state_dict, strength_model, strength_clip, **extra
    )


_ROTATION_INPUT = (
    [core.ROTATION_AUTO, *core.ROTATIONS],
    {
        "default": core.ROTATION_AUTO,
        "tooltip": "auto quantizes with and without a ConvRot (Hadamard) rotation "
        "and keeps whichever has the lower SSE. none and convrot force one choice.",
    },
)


class INT8LoRAQuantize:
    """Convert a regular LoRA/LoKr from the loras folder into an INT8 LoRA."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (
                    folder_paths.get_filename_list("loras"),
                    {"tooltip": "Source LoRA from the regular loras folder."},
                ),
                "output_name": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "File name to save under models/loras_int8. "
                        "Leave empty to use '<source name>_int8'.",
                    },
                ),
                "rotation": _ROTATION_INPUT,
                "overwrite": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Replace the output file if it exists."},
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "quantize"
    OUTPUT_NODE = True
    CATEGORY = "loaders/INT8 LoRA"
    DESCRIPTION = (
        "Quantizes a LoRA to INT8 and reports the size change and the SSE "
        "between the original and the INT8 version. Connect 'report' to a "
        "Preview Any node to see it in the graph; it is also printed to the "
        "console."
    )

    def quantize(self, lora_name, output_name, rotation, overwrite):
        source = folder_paths.get_full_path_or_raise("loras", lora_name)
        stem = os.path.splitext(os.path.basename(lora_name))[0]
        file_name = core.sanitize_output_name(output_name, stem)
        target = _int8_target(file_name)

        if os.path.exists(target) and not overwrite:
            raise FileExistsError(
                f"{file_name} already exists in models/{INT8_FOLDER}; "
                "enable 'overwrite' or choose another output_name."
            )

        report = _quantize_to_file(source, target, rotation)
        return _report_result(core.format_report(report, lora_name, file_name))


class INT8LoRAQuantizeFolder:
    """Quantize every LoRA under a subfolder of the loras folder."""

    @classmethod
    def INPUT_TYPES(cls):
        folders = core.lora_subfolders(folder_paths.get_filename_list("loras"))
        return {
            "required": {
                "folder": (
                    folders or [NO_FOLDERS],
                    {"tooltip": "Subfolder of the loras folder. Everything under it, "
                     "including nested folders, is processed."},
                ),
                "rotation": _ROTATION_INPUT,
                "overwrite": (
                    "BOOLEAN",
                    {"default": False,
                     "tooltip": "Replace INT8 files that already exist. "
                     "Off: those files are skipped."},
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "quantize_folder"
    OUTPUT_NODE = True
    CATEGORY = "loaders/INT8 LoRA"
    DESCRIPTION = (
        "Quantizes every LoRA under a loras subfolder to INT8, keeping the folder "
        "structure under models/loras_int8. Files that are not LoRAs, are already "
        "INT8, already have an output, or cannot be read are skipped and listed "
        "in the report."
    )

    def quantize_folder(self, folder, rotation, overwrite):
        names = core.select_in_folder(folder_paths.get_filename_list("loras"), folder)
        if not names:
            raise ValueError(f"No LoRA files found under '{folder}'.")
        progress = comfy.utils.ProgressBar(len(names))

        def process(name):
            comfy.model_management.throw_exception_if_processing_interrupted()
            target = _int8_target(core.int8_relative_path(name))
            if os.path.exists(target) and not overwrite:
                raise FileExistsError("output already exists (enable overwrite to replace)")
            source = folder_paths.get_full_path_or_raise("loras", name)
            keys = core.safetensors_keys(source)
            if keys is None:
                raise ValueError("not a readable .safetensors file")
            core.check_batch_candidate(keys)
            return _quantize_to_file(source, target, rotation)

        entries = core.run_batch(names, process, on_progress=lambda: progress.update(1))
        return _report_result(core.format_batch_report(folder, entries))


_STRENGTH = {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}


class INT8LoRALoader:
    """Apply one INT8 LoRA to a model only."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list(INT8_FOLDER),),
                "strength": ("FLOAT", _STRENGTH),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "loaders/INT8 LoRA"
    DESCRIPTION = "Applies one INT8 LoRA from models/loras_int8 to the model."

    def load(self, model, lora_name, strength):
        if strength == 0:
            return (model,)
        model, _ = _apply_int8_lora(model, None, lora_name, strength, 0.0)
        return (model,)


class INT8LoRALoaderModelClip:
    """Apply one INT8 LoRA to a model and a CLIP."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list(INT8_FOLDER),),
                "strength_model": ("FLOAT", _STRENGTH),
                "strength_clip": ("FLOAT", _STRENGTH),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "load"
    CATEGORY = "loaders/INT8 LoRA"
    DESCRIPTION = "Applies one INT8 LoRA from models/loras_int8 to the model and CLIP."

    def load(self, model, clip, lora_name, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)
        return _apply_int8_lora(model, clip, lora_name, strength_model, strength_clip)


class INT8LoRALoaderStack:
    """Apply up to three INT8 LoRAs to a model only."""

    @classmethod
    def INPUT_TYPES(cls):
        choices = _int8_choices()
        optional = {}
        for i in range(1, STACK_SLOTS + 1):
            optional[f"lora_{i}"] = (choices, {"default": NONE_CHOICE})
            optional[f"strength_{i}"] = ("FLOAT", _STRENGTH)
        return {"required": {"model": ("MODEL",)}, "optional": optional}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "loaders/INT8 LoRA"
    DESCRIPTION = (
        "Applies up to three INT8 LoRAs from models/loras_int8 to the model. "
        "Slots set to None are skipped."
    )

    def load(self, model, **kwargs):
        for i in range(1, STACK_SLOTS + 1):
            name = kwargs.get(f"lora_{i}", NONE_CHOICE)
            strength = kwargs.get(f"strength_{i}", 1.0)
            if name == NONE_CHOICE or strength == 0:
                continue
            model, _ = _apply_int8_lora(model, None, name, strength, 0.0)
        return (model,)


NODE_CLASS_MAPPINGS = {
    "INT8LoRAQuantize": INT8LoRAQuantize,
    "INT8LoRAQuantizeFolder": INT8LoRAQuantizeFolder,
    "INT8LoRALoader": INT8LoRALoader,
    "INT8LoRALoaderModelClip": INT8LoRALoaderModelClip,
    "INT8LoRALoaderStack": INT8LoRALoaderStack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "INT8LoRAQuantize": "Quantize LoRA to INT8",
    "INT8LoRAQuantizeFolder": "Quantize LoRA Folder to INT8",
    "INT8LoRALoader": "Load INT8 LoRA",
    "INT8LoRALoaderModelClip": "Load INT8 LoRA (Model + CLIP)",
    "INT8LoRALoaderStack": "Load INT8 LoRA Stack (3, Model only)",
}
