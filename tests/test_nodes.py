"""Tests for nodes.py, run against the stand-in ComfyUI in fake_comfy.py."""

import hashlib
import math
import os
import unittest
from unittest import mock

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from fake_comfy import FakeComfy, FakeInterrupt, FakeInterruptException
from test_int8_lora_core import make_lora

A = os.path.join("styles", "a.safetensors")  # bf16, has metadata
B = os.path.join("styles", "deep", "b.safetensors")  # fp32, no metadata
DTYPE_CHOICES = ["original", "bf16", "fp16", "fp32"]


def write_loras(fake, with_subfolders=True):
    save_file(make_lora(), os.path.join(fake.loras_dir, "root.safetensors"))
    if not with_subfolders:
        return
    styles = os.path.join(fake.loras_dir, "styles")
    os.makedirs(os.path.join(styles, "deep"))
    save_file(make_lora(torch.bfloat16), os.path.join(styles, "a.safetensors"), metadata={"ss_x": "1"})
    save_file(make_lora(torch.float32), os.path.join(styles, "deep", "b.safetensors"))
    save_file({"model.diffusion_model.w": torch.randn(64, 64)}, os.path.join(styles, "checkpoint.safetensors"))
    torch.save({"w": torch.ones(2, 2)}, os.path.join(styles, "old.pt"))


def write_nan_lora(path):
    """A LoRA-looking file whose values cannot be quantized (two separate tensors)."""
    save_file(
        {
            "x.lora_up.weight": torch.full((64, 64), float("nan")),
            "x.lora_down.weight": torch.full((64, 64), float("nan")),
        },
        path,
    )


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def file_metadata(path):
    with safe_open(path, framework="pt", device="cpu") as f:
        return dict(f.metadata() or {})


class NodeTestCase(unittest.TestCase):
    accepts_lora_metadata = True
    interrupt_is_exception = False

    def setUp(self):
        self.fake = FakeComfy(
            accepts_lora_metadata=self.accepts_lora_metadata,
            interrupt_is_exception=self.interrupt_is_exception,
        )
        self.addCleanup(self.fake.cleanup)
        self.nodes = self.fake.nodes
        write_loras(self.fake)

    def quantize(self, source, output_name="", rotation="none", overwrite=False):
        return self.nodes.INT8LoRAQuantize().quantize(source, output_name, rotation, overwrite)

    def quantize_both(self):
        self.quantize(A, "a_int8")
        self.quantize(B, "b_int8")


class RegistrationTests(NodeTestCase):
    def test_all_nodes_are_registered_with_display_names(self):
        expected = {
            "INT8LoRAQuantize", "INT8LoRAQuantizeFolder", "INT8LoRALoader",
            "INT8LoRALoaderModelClip", "INT8LoRALoaderStack", "INT8LoRASwitch",
        }
        self.assertEqual(set(self.nodes.NODE_CLASS_MAPPINGS), expected)
        self.assertEqual(set(self.nodes.NODE_DISPLAY_NAME_MAPPINGS), expected)

    def test_int8_folder_is_created_and_only_lists_safetensors(self):
        self.assertTrue(os.path.isdir(self.fake.int8_dir))
        for name in ("ok.safetensors", "junk.txt", "half.safetensors.tmp"):
            open(os.path.join(self.fake.int8_dir, name), "wb").close()
        listed = self.fake.folder_paths.get_filename_list("loras_int8")
        self.assertEqual(listed, ["ok.safetensors"])

    def test_loaders_have_dtype_as_an_optional_input(self):
        # Optional, so workflows saved before the option existed still validate.
        for cls in (self.nodes.INT8LoRALoader, self.nodes.INT8LoRALoaderModelClip, self.nodes.INT8LoRALoaderStack):
            spec = cls.INPUT_TYPES()
            self.assertNotIn("dtype", spec["required"], cls.__name__)
            choices, options = spec["optional"]["dtype"]
            self.assertEqual(choices, DTYPE_CHOICES, cls.__name__)
            self.assertEqual(options["default"], "bf16")

    def test_loader_inputs(self):
        single = self.nodes.INT8LoRALoader.INPUT_TYPES()
        self.assertEqual(list(single["required"]), ["model", "lora_name", "strength"])
        both = self.nodes.INT8LoRALoaderModelClip.INPUT_TYPES()
        self.assertEqual(list(both["required"]), ["model", "clip", "lora_name", "strength_model", "strength_clip"])
        stack = self.nodes.INT8LoRALoaderStack.INPUT_TYPES()
        self.assertEqual(list(stack["required"]), ["model"])
        self.assertEqual(
            list(stack["optional"]),
            ["lora_1", "strength_1", "lora_2", "strength_2", "lora_3", "strength_3", "dtype"],
        )
        self.assertEqual(stack["optional"]["lora_1"][0][0], "None")

    def test_quantizers_share_the_rotation_choices(self):
        for cls in (self.nodes.INT8LoRAQuantize, self.nodes.INT8LoRAQuantizeFolder):
            choices, options = cls.INPUT_TYPES()["required"]["rotation"]
            self.assertEqual(choices, ["auto", "none", "convrot"])
            self.assertEqual(options["default"], "auto")

    def test_report_result_shape(self):
        self.assertEqual(self.nodes._report_result("hi"), {"ui": {"text": ["hi"]}, "result": ("hi",)})

    def test_output_path_cannot_escape_the_int8_folder(self):
        inside = self.nodes._int8_target("styles/a_int8.safetensors")
        self.assertEqual(inside, os.path.join(os.path.abspath(self.fake.int8_dir), "styles", "a_int8.safetensors"))
        for bad in ("../evil.safetensors", "styles/../../evil.safetensors"):
            with self.assertRaises(ValueError):
                self.nodes._int8_target(bad)


class QuantizeNodeTests(NodeTestCase):
    def test_writes_the_int8_file_and_returns_the_report(self):
        result = self.quantize(A, "")
        self.assertEqual(self.fake.int8_files(), ["a_int8.safetensors"])  # default name
        text = result["result"][0]
        self.assertEqual(result["ui"], {"text": [text]})
        self.assertIn("MiB", text)
        self.assertIn("rotation: none", text)

    def test_output_name_is_used(self):
        self.quantize(A, "mine")
        self.assertEqual(self.fake.int8_files(), ["mine.safetensors"])

    def test_source_file_is_never_modified(self):
        path = os.path.join(self.fake.loras_dir, A)
        before = sha256(path)
        self.quantize(A)
        self.assertEqual(sha256(path), before)

    def test_source_metadata_is_copied_into_the_output(self):
        self.quantize(A, "a_int8")
        meta = file_metadata(os.path.join(self.fake.int8_dir, "a_int8.safetensors"))
        self.assertEqual(meta["ss_x"], "1")
        self.assertEqual(meta["int8_lora_format"], "1")

    def test_existing_output_needs_overwrite(self):
        self.quantize(A, "out")
        with self.assertRaises(FileExistsError):
            self.quantize(A, "out")
        self.quantize(A, "out", overwrite=True)  # allowed when asked

    def test_unsafe_output_names_are_rejected(self):
        for bad in ("../evil", "a/b", ".."):
            with self.assertRaises(ValueError):
                self.quantize(A, bad)
        self.assertEqual(self.fake.int8_files(), [])

    def test_rotation_choice_reaches_the_file(self):
        for choice, allowed in (("none", {"none"}), ("convrot", {"convrot"}), ("auto", {"none", "convrot"})):
            self.quantize(A, f"r_{choice}", rotation=choice)
            meta = file_metadata(os.path.join(self.fake.int8_dir, f"r_{choice}.safetensors"))
            self.assertIn(meta["int8_lora_rotation"], allowed, choice)

    def test_failure_while_writing_leaves_no_files_behind(self):
        def broken_save(path, tensors, metadata):
            with open(path, "wb") as f:
                f.write(b"partial")
            raise OSError("disk full")

        with mock.patch.object(self.nodes.core, "save_int8_lora", broken_save):
            with self.assertRaises(OSError):
                self.quantize(A, "out")
        self.assertEqual(self.fake.int8_files(), [])  # no output and no .tmp

    def test_bad_source_values_fail_cleanly(self):
        write_nan_lora(os.path.join(self.fake.loras_dir, "bad.safetensors"))
        with self.assertRaisesRegex(ValueError, "NaN"):
            self.quantize("bad.safetensors")
        self.assertEqual(self.fake.int8_files(), [])


class LoaderTests(NodeTestCase):
    def setUp(self):
        super().setUp()
        self.quantize_both()
        self.last = lambda: self.fake.applied[-1]

    def test_dropdown_lists_the_int8_files(self):
        listed = self.nodes.INT8LoRALoader.INPUT_TYPES()["required"]["lora_name"][0]
        self.assertEqual(listed, ["a_int8.safetensors", "b_int8.safetensors"])

    def test_default_dtype_is_bf16(self):
        loader = self.nodes.INT8LoRALoader()
        self.assertEqual(loader.load("M", "a_int8.safetensors", 0.8), ("M+L1",))
        self.assertEqual(self.last()["dtypes"], {torch.bfloat16})
        self.assertEqual((self.last()["strength_model"], self.last()["strength_clip"]), (0.8, 0.0))
        self.assertIsNone(self.last()["clip_in"])
        loader.load("M", "b_int8.safetensors", 1.0)  # an fp32 LoRA is restored as bf16 by default
        self.assertEqual(self.last()["dtypes"], {torch.bfloat16})

    def test_every_loader_defaults_to_bf16_for_an_fp32_lora(self):
        self.nodes.INT8LoRALoader().load("M", "b_int8.safetensors", 1.0)
        self.nodes.INT8LoRALoaderModelClip().load("M", "C", "b_int8.safetensors", 1.0, 1.0)
        self.nodes.INT8LoRALoaderStack().load("M", lora_1="b_int8.safetensors", strength_1=1.0)
        self.assertEqual([a["dtypes"] for a in self.fake.applied], [{torch.bfloat16}] * 3)

    def test_every_dtype_choice_is_applied(self):
        loader = self.nodes.INT8LoRALoader()
        expected = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        for source in ("a_int8.safetensors", "b_int8.safetensors"):  # bf16 and fp32 sources
            for name, dtype in expected.items():
                loader.load("M", source, 1.0, name)
                self.assertEqual(self.last()["dtypes"], {dtype}, (source, name))

    def test_original_choice_keeps_the_stored_dtype(self):
        loader = self.nodes.INT8LoRALoader()
        loader.load("M", "b_int8.safetensors", 1.0, "original")
        self.assertEqual(self.last()["dtypes"], {torch.float32})

    def test_the_applied_state_dict_is_complete_and_plain(self):
        self.nodes.INT8LoRALoader().load("M", "a_int8.safetensors", 1.0, "fp16")
        self.assertEqual(self.last()["keys"], set(make_lora()))  # no *.int8_scale keys left

    def test_zero_strength_skips_the_load(self):
        self.assertEqual(self.nodes.INT8LoRALoader().load("M", "a_int8.safetensors", 0), ("M",))
        self.assertEqual(self.fake.applied, [])

    def test_unknown_dtype_and_missing_file_are_errors(self):
        loader = self.nodes.INT8LoRALoader()
        with self.assertRaises(ValueError):
            loader.load("M", "a_int8.safetensors", 1.0, "fp8")
        with self.assertRaises(FileNotFoundError):
            loader.load("M", "nope.safetensors", 1.0)

    def test_model_clip_loader_passes_both_strengths_and_dtype(self):
        node = self.nodes.INT8LoRALoaderModelClip()
        self.assertEqual(node.load("M", "C", "a_int8.safetensors", 0.5, -0.25, "fp16"), ("M+L1", "C+L1"))
        applied = self.last()
        self.assertEqual((applied["strength_model"], applied["strength_clip"]), (0.5, -0.25))
        self.assertEqual((applied["model_in"], applied["clip_in"]), ("M", "C"))
        self.assertEqual(applied["dtypes"], {torch.float16})

    def test_model_clip_loader_skips_only_when_both_strengths_are_zero(self):
        node = self.nodes.INT8LoRALoaderModelClip()
        self.assertEqual(node.load("M", "C", "a_int8.safetensors", 0, 0), ("M", "C"))
        self.assertEqual(self.fake.applied, [])
        node.load("M", "C", "a_int8.safetensors", 0, 0.5)
        self.assertEqual(len(self.fake.applied), 1)

    def test_stack_applies_slots_in_order_and_skips_empty_ones(self):
        out = self.nodes.INT8LoRALoaderStack().load(
            "M", lora_1="a_int8.safetensors", strength_1=1.0, lora_2="None", strength_2=1.0,
            lora_3="b_int8.safetensors", strength_3=0.5, dtype="bf16",
        )
        self.assertEqual(out, ("M+L1+L2",))
        self.assertEqual([a["strength_model"] for a in self.fake.applied], [1.0, 0.5])
        self.assertEqual([a["dtypes"] for a in self.fake.applied], [{torch.bfloat16}, {torch.bfloat16}])

    def test_stack_skips_zero_strength_and_defaults_the_dtype(self):
        out = self.nodes.INT8LoRALoaderStack().load(
            "M", lora_1="a_int8.safetensors", strength_1=0, lora_2="b_int8.safetensors", strength_2=1.0,
        )
        self.assertEqual(out, ("M+L1",))
        self.assertEqual(self.last()["dtypes"], {torch.bfloat16})  # b is fp32, dtype omitted: default bf16

    def test_source_metadata_is_handed_to_comfy(self):
        loader = self.nodes.INT8LoRALoader()
        loader.load("M", "a_int8.safetensors", 1.0)
        self.assertEqual(self.last()["metadata"], {"ss_x": "1"})
        loader.load("M", "b_int8.safetensors", 1.0)
        self.assertIsNone(self.last()["metadata"])  # nothing to pass on


class SwitchTests(NodeTestCase):
    def setUp(self):
        super().setUp()
        self.quantize(A)  # a flat INT8 copy: a_int8.safetensors
        self.switch = self.nodes.INT8LoRASwitch()
        self.fake.torch_file_loads.clear()  # count only what the switch reads

    def test_inputs(self):
        spec = self.nodes.INT8LoRASwitch.INPUT_TYPES()
        self.assertEqual(list(spec["required"]), ["model", "lora_name", "mode", "strength"])
        modes, options = spec["required"]["mode"]
        self.assertEqual(modes, ["int8", "original", "none"])
        self.assertEqual(options["default"], "int8")
        self.assertEqual(list(spec["optional"]), ["clip", "strength_clip", "dtype"])
        names = spec["required"]["lora_name"][0]  # the NORMAL loras, not the INT8 files
        self.assertIn("root.safetensors", names)
        self.assertIn(A, names)
        self.assertNotIn("a_int8.safetensors", names)

    def test_int8_mode_applies_the_int8_copy(self):
        out = self.switch.switch("M", A, "int8", 0.8, dtype="fp32")
        self.assertEqual(out, ("M+L1", None, "INT8 file: a_int8.safetensors (restored as fp32)"))
        applied = self.fake.applied[-1]
        self.assertEqual(applied["dtypes"], {torch.float32})  # the dtype choice is honoured
        self.assertEqual(applied["strength_model"], 0.8)
        self.assertEqual(applied["metadata"], {"ss_x": "1"})
        self.assertEqual(self.fake.torch_file_loads, [])  # the original file was not read

    def test_original_mode_applies_the_normal_file_and_ignores_dtype(self):
        out = self.switch.switch("M", A, "original", 0.8, dtype="fp32")
        self.assertEqual(out, ("M+L1", None, f"original file: {A}"))
        applied = self.fake.applied[-1]
        self.assertEqual(applied["dtypes"], {torch.bfloat16})  # the file's own dtype
        self.assertEqual(applied["keys"], set(make_lora()))
        self.assertEqual(applied["metadata"], {"ss_x": "1"})
        self.assertEqual(self.fake.torch_file_loads, [os.path.join(self.fake.loras_dir, A)])

    def test_both_modes_hand_comfy_the_same_kind_of_lora(self):
        self.switch.switch("M", A, "int8", 0.5)
        self.switch.switch("M", A, "original", 0.5)
        first, second = self.fake.applied
        self.assertEqual(first["keys"], second["keys"])
        self.assertEqual(first["dtypes"], second["dtypes"])  # default dtype restores bf16
        self.assertEqual(first["strength_model"], second["strength_model"])

    def test_int8_mode_defaults_to_bf16_for_an_fp32_lora(self):
        self.quantize(B, "b_int8")
        self.switch.switch("M", B, "int8", 1.0)
        self.assertEqual(self.fake.applied[-1]["dtypes"], {torch.bfloat16})
        self.switch.switch("M", B, "int8", 1.0, dtype="original")  # asking for the stored dtype
        self.assertEqual(self.fake.applied[-1]["dtypes"], {torch.float32})

    def test_none_mode_applies_nothing(self):
        out = self.switch.switch("M", A, "none", 1.0, clip="C")
        self.assertEqual(out, ("M", "C", "no LoRA applied (mode: none)"))
        self.assertEqual(self.fake.applied, [])

    def test_clip_is_patched_when_connected(self):
        out = self.switch.switch("M", A, "int8", 0.5, clip="C", strength_clip=-0.25)
        self.assertEqual(out[:2], ("M+L1", "C+L1"))
        applied = self.fake.applied[-1]
        self.assertEqual((applied["strength_model"], applied["strength_clip"]), (0.5, -0.25))
        self.assertEqual(applied["clip_in"], "C")

    def test_without_a_clip_the_clip_strength_is_zero(self):
        self.switch.switch("M", A, "int8", 0.5, strength_clip=0.9)
        applied = self.fake.applied[-1]
        self.assertIsNone(applied["clip_in"])
        self.assertEqual(applied["strength_clip"], 0.0)

    def test_zero_strength_applies_nothing(self):
        out = self.switch.switch("M", A, "int8", 0)
        self.assertEqual(out, ("M", None, "no LoRA applied (strength 0)"))
        self.assertEqual(self.fake.applied, [])
        self.switch.switch("M", A, "int8", 0, clip="C", strength_clip=0.5)  # clip strength still counts
        self.assertEqual(len(self.fake.applied), 1)

    def test_a_missing_int8_copy_gives_a_helpful_error(self):
        with self.assertRaises(FileNotFoundError) as caught:
            self.switch.switch("M", B, "int8", 1.0)
        message = str(caught.exception)
        self.assertIn("Quantize LoRA to INT8", message)
        self.assertIn("b_int8.safetensors", message)
        self.assertEqual(self.fake.applied, [])
        self.switch.switch("M", B, "original", 1.0)  # the original still works
        self.assertEqual(len(self.fake.applied), 1)

    def test_the_mirrored_copy_is_preferred_over_the_flat_one(self):
        self.nodes.INT8LoRAQuantizeFolder().quantize_folder("styles", "none", False)
        out = self.switch.switch("M", A, "int8", 1.0)
        self.assertIn("styles/a_int8.safetensors", out[2].replace("\\", "/"))


class OlderComfyTests(NodeTestCase):
    accepts_lora_metadata = False

    def test_metadata_argument_is_skipped_when_comfy_does_not_accept_it(self):
        self.quantize(A, "a_int8")
        self.nodes.INT8LoRALoader().load("M", "a_int8.safetensors", 1.0, "bf16")
        self.assertEqual(self.fake.applied[-1]["metadata"], "<no argument>")
        self.assertEqual(self.fake.applied[-1]["dtypes"], {torch.bfloat16})


class FolderNodeTests(NodeTestCase):
    def run_folder(self, folder="styles", rotation="none", overwrite=False):
        return self.nodes.INT8LoRAQuantizeFolder().quantize_folder(folder, rotation, overwrite)["result"][0]

    def test_dropdown_lists_subfolders_at_every_depth(self):
        choices = self.nodes.INT8LoRAQuantizeFolder.INPUT_TYPES()["required"]["folder"][0]
        self.assertEqual(choices, ["styles", "styles/deep"])

    def test_dropdown_has_a_placeholder_without_subfolders(self):
        fake = FakeComfy()
        self.addCleanup(fake.cleanup)
        write_loras(fake, with_subfolders=False)
        choices = fake.nodes.INT8LoRAQuantizeFolder.INPUT_TYPES()["required"]["folder"][0]
        self.assertEqual(choices, [fake.nodes.NO_FOLDERS])
        with self.assertRaisesRegex(ValueError, "No LoRA files"):
            fake.nodes.INT8LoRAQuantizeFolder().quantize_folder(fake.nodes.NO_FOLDERS, "none", False)

    def test_quantizes_everything_below_the_folder_and_mirrors_it(self):
        text = self.run_folder()
        self.assertEqual(self.fake.int8_files(), ["styles/a_int8.safetensors", "styles/deep/b_int8.safetensors"])
        self.assertIn("2 quantized, 2 skipped", text)
        self.assertIn("does not look like a LoRA", text)  # the checkpoint
        self.assertIn("not a readable .safetensors file", text)  # the .pt file
        self.assertEqual(self.fake.progress, [1, 1, 1, 1])  # one tick per file, skipped or not

    def test_root_files_and_other_folders_are_left_alone(self):
        self.run_folder()
        self.assertNotIn("root_int8.safetensors", self.fake.int8_files())

    def test_only_the_selected_nested_folder_is_processed(self):
        text = self.run_folder("styles/deep")
        self.assertEqual(self.fake.int8_files(), ["styles/deep/b_int8.safetensors"])
        self.assertIn("1 quantized, 0 skipped", text)

    def test_rerun_skips_existing_outputs_until_overwrite_is_on(self):
        self.run_folder()
        self.assertIn("0 quantized, 4 skipped", self.run_folder())
        self.assertIn("output already exists", self.run_folder())
        self.assertIn("2 quantized", self.run_folder(overwrite=True))

    def test_files_are_written_with_the_chosen_rotation(self):
        self.run_folder(rotation="convrot")
        for name in self.fake.int8_files():
            meta = file_metadata(os.path.join(self.fake.int8_dir, *name.split("/")))
            self.assertEqual(meta["int8_lora_rotation"], "convrot", name)

    def test_one_bad_file_does_not_stop_the_batch(self):
        write_nan_lora(os.path.join(self.fake.loras_dir, "styles", "bad.safetensors"))
        text = self.run_folder()
        self.assertIn("2 quantized, 3 skipped", text)
        self.assertIn("NaN or Inf", text)
        self.assertEqual(self.fake.int8_files(), ["styles/a_int8.safetensors", "styles/deep/b_int8.safetensors"])

    def test_cancelling_stops_the_batch_but_keeps_finished_files(self):
        self.fake.interrupt_on_call = 2  # the second file triggers the cancel
        with self.assertRaises(FakeInterrupt):
            self.run_folder()
        self.assertEqual(self.fake.int8_files(), ["styles/a_int8.safetensors"])

    def test_unknown_folder_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "No LoRA files"):
            self.run_folder("nope")
        with self.assertRaisesRegex(ValueError, "No LoRA files"):
            self.run_folder("style")  # a partial name is not a folder


class OlderComfyCancelTests(NodeTestCase):
    # Older ComfyUI derives its cancel exception from Exception, not BaseException.
    interrupt_is_exception = True

    def test_cancelling_still_stops_the_folder_batch(self):
        self.fake.interrupt_on_call = 2  # the second file triggers the cancel
        with self.assertRaises(FakeInterruptException):
            self.nodes.INT8LoRAQuantizeFolder().quantize_folder("styles", "none", False)
        self.assertEqual(self.fake.int8_files(), ["styles/a_int8.safetensors"])


class CachingTests(NodeTestCase):
    """IS_CHANGED must change when the files involved change, and only then."""

    def single(self, source=A, name="", rotation="none", overwrite=False):
        return self.nodes.INT8LoRAQuantize.IS_CHANGED(source, name, rotation, overwrite)

    def folder(self, folder="styles"):
        return self.nodes.INT8LoRAQuantizeFolder.IS_CHANGED(folder, "none", False)

    def test_single_is_stable_while_nothing_changes(self):
        self.assertEqual(self.single(), self.single())
        self.quantize(A)
        self.assertEqual(self.single(), self.single())

    def test_single_reruns_when_the_output_appears_or_disappears(self):
        before = self.single()
        self.quantize(A)
        made = self.single()
        self.assertNotEqual(before, made)
        os.remove(os.path.join(self.fake.int8_dir, "a_int8.safetensors"))
        self.assertEqual(self.single(), before)  # missing again: same state as at the start

    def test_single_reruns_when_the_source_is_edited(self):
        before = self.single()
        save_file(make_lora(torch.float32), os.path.join(self.fake.loras_dir, A))  # different size
        self.assertNotEqual(self.single(), before)

    def test_single_depends_on_the_output_name_and_ignores_unrelated_files(self):
        self.assertNotEqual(self.single(name="one"), self.single(name="two"))
        before = self.single()
        save_file(make_lora(), os.path.join(self.fake.loras_dir, "unrelated.safetensors"))
        self.assertEqual(self.single(), before)

    def test_single_falls_back_to_always_run_when_it_cannot_tell(self):
        self.assertTrue(math.isnan(self.single(source="missing.safetensors")))
        self.assertTrue(math.isnan(self.single(name="../evil")))

    def test_folder_is_stable_while_nothing_changes(self):
        self.assertEqual(self.folder(), self.folder())

    def test_folder_reruns_when_a_lora_is_added(self):
        before = self.folder()
        save_file(make_lora(), os.path.join(self.fake.loras_dir, "styles", "new.safetensors"))
        self.assertNotEqual(self.folder(), before)

    def test_folder_reruns_after_a_run_and_when_an_output_is_deleted(self):
        before = self.folder()
        self.nodes.INT8LoRAQuantizeFolder().quantize_folder("styles", "none", False)
        after_run = self.folder()
        self.assertNotEqual(after_run, before)
        self.assertEqual(self.folder(), after_run)  # nothing changed since: cached
        os.remove(os.path.join(self.fake.int8_dir, "styles", "a_int8.safetensors"))
        self.assertNotEqual(self.folder(), after_run)

    def test_folder_ignores_files_outside_it(self):
        before = self.folder("styles/deep")
        save_file(make_lora(), os.path.join(self.fake.loras_dir, "styles", "new.safetensors"))
        self.assertEqual(self.folder("styles/deep"), before)  # a sibling file, not under deep
        save_file(make_lora(), os.path.join(self.fake.loras_dir, "root2.safetensors"))
        self.assertEqual(self.folder("styles/deep"), before)

    def test_folder_with_no_files_still_gives_a_value(self):
        self.assertIsInstance(self.folder("nope"), str)  # the run itself reports the problem


if __name__ == "__main__":
    unittest.main()
