"""Stand-in ComfyUI modules, so nodes.py can be imported and exercised in tests.

The real modules only exist inside a running ComfyUI. These fakes implement just
the functions nodes.py calls, with the same names and signatures, and record what
happens (which LoRA was applied with which dtype, progress ticks, ...).
"""

import importlib.util
import itertools
import os
import shutil
import sys
import tempfile
import types

import torch
from safetensors.torch import load_file

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_FAKE_MODULES = ("folder_paths", "comfy", "comfy.sd", "comfy.utils", "comfy.model_management")
_counter = itertools.count(1)


class FakeInterrupt(BaseException):
    """Like ComfyUI's InterruptProcessingException in current versions: a BaseException."""


class FakeInterruptException(Exception):
    """Like InterruptProcessingException in older ComfyUI versions: an Exception."""


class FakeComfy:
    """A fresh fake ComfyUI install in a temp folder, with the pack imported."""

    def __init__(self, accepts_lora_metadata: bool = True, interrupt_is_exception: bool = False):
        self.root = tempfile.mkdtemp()
        self.models_dir = os.path.join(self.root, "models")
        self.loras_dir = os.path.join(self.models_dir, "loras")
        os.makedirs(self.loras_dir)
        self.applied = []  # one dict per load_lora_for_models call
        self.progress = []  # ProgressBar.update values
        self.torch_file_loads = []  # paths read through comfy.utils.load_torch_file
        self.interrupt_on_call = None  # raise FakeInterrupt on this numbered call
        self._interrupt_calls = 0
        self.interrupt_class = FakeInterruptException if interrupt_is_exception else FakeInterrupt
        self._install(accepts_lora_metadata)
        self.nodes = self._import_pack()

    # ------------------------------------------------------------ fake modules
    def _install(self, accepts_lora_metadata: bool) -> None:
        outer = self

        fp = types.ModuleType("folder_paths")
        fp.models_dir = self.models_dir
        fp.folder_names_and_paths = {"loras": ([self.loras_dir], {".safetensors", ".pt"})}

        def add_model_folder_path(name, path, is_default=False):
            if name in fp.folder_names_and_paths:
                paths = fp.folder_names_and_paths[name][0]
                if path not in paths:
                    paths.insert(0, path) if is_default else paths.append(path)
            else:
                fp.folder_names_and_paths[name] = ([path], set())

        def get_folder_paths(name):
            return list(fp.folder_names_and_paths[name][0])

        def get_filename_list(name):
            bases, extensions = fp.folder_names_and_paths[name]
            found = []
            for base in bases:
                for dirpath, _, files in os.walk(base):
                    for file in files:
                        if not extensions or os.path.splitext(file)[1].lower() in extensions:
                            found.append(os.path.relpath(os.path.join(dirpath, file), base))
            return sorted(found)

        def get_full_path_or_raise(name, filename):
            for base in fp.folder_names_and_paths[name][0]:
                path = os.path.join(base, filename)
                if os.path.isfile(path):
                    return path
            raise FileNotFoundError(f"Model in folder '{name}' with filename '{filename}' not found.")

        fp.add_model_folder_path = add_model_folder_path
        fp.get_folder_paths = get_folder_paths
        fp.get_filename_list = get_filename_list
        fp.get_full_path_or_raise = get_full_path_or_raise

        def record(model, clip, lora, strength_model, strength_clip, metadata):
            weights = {k: v for k, v in lora.items() if k.endswith(("lora_up.weight", "lokr_w2"))}
            outer.applied.append(
                {
                    "model_in": model,
                    "clip_in": clip,
                    "strength_model": strength_model,
                    "strength_clip": strength_clip,
                    "metadata": metadata,
                    "dtypes": {v.dtype for v in weights.values()},
                    "keys": set(lora),
                }
            )
            tag = f"+L{len(outer.applied)}"
            return (None if model is None else model + tag), (None if clip is None else clip + tag)

        comfy_sd = types.ModuleType("comfy.sd")
        if accepts_lora_metadata:
            def load_lora_for_models(model, clip, lora, strength_model, strength_clip, lora_metadata=None):
                return record(model, clip, lora, strength_model, strength_clip, lora_metadata)
        else:
            def load_lora_for_models(model, clip, lora, strength_model, strength_clip):
                return record(model, clip, lora, strength_model, strength_clip, "<no argument>")
        comfy_sd.load_lora_for_models = load_lora_for_models

        comfy_utils = types.ModuleType("comfy.utils")

        def load_torch_file(path, safe_load=False):
            outer.torch_file_loads.append(path)
            if path.endswith(".safetensors"):
                return load_file(path)
            return torch.load(path, weights_only=True)

        class ProgressBar:
            def __init__(self, total):
                self.total = total

            def update(self, value):
                outer.progress.append(value)

        comfy_utils.load_torch_file = load_torch_file
        comfy_utils.ProgressBar = ProgressBar

        comfy_mm = types.ModuleType("comfy.model_management")

        def throw_exception_if_processing_interrupted():
            outer._interrupt_calls += 1
            if outer.interrupt_on_call == outer._interrupt_calls:
                raise outer.interrupt_class()

        comfy_mm.InterruptProcessingException = outer.interrupt_class
        comfy_mm.throw_exception_if_processing_interrupted = throw_exception_if_processing_interrupted

        comfy = types.ModuleType("comfy")
        comfy.sd, comfy.utils, comfy.model_management = comfy_sd, comfy_utils, comfy_mm
        modules = {
            "folder_paths": fp, "comfy": comfy, "comfy.sd": comfy_sd,
            "comfy.utils": comfy_utils, "comfy.model_management": comfy_mm,
        }
        sys.modules.update(modules)
        self.folder_paths = fp

    def _import_pack(self):
        """Import the repo as a package, the way ComfyUI imports a custom node."""
        self.package = f"int8pack_{next(_counter)}"
        spec = importlib.util.spec_from_file_location(
            self.package, os.path.join(REPO, "__init__.py"), submodule_search_locations=[REPO]
        )
        pack = importlib.util.module_from_spec(spec)
        sys.modules[self.package] = pack
        spec.loader.exec_module(pack)
        return sys.modules[f"{self.package}.nodes"]

    # ------------------------------------------------------------------ helpers
    @property
    def int8_dir(self):
        return os.path.join(self.models_dir, "loras_int8")

    def int8_files(self):
        """Relative paths (with '/') of everything under the INT8 folder."""
        return sorted(
            os.path.relpath(os.path.join(d, f), self.int8_dir).replace(os.sep, "/")
            for d, _, files in os.walk(self.int8_dir)
            for f in files
        )

    def cleanup(self):
        for name in list(sys.modules):
            if name in _FAKE_MODULES or name.startswith(self.package):
                del sys.modules[name]
        shutil.rmtree(self.root, ignore_errors=True)
