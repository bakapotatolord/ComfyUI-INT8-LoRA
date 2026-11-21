import os
import pathlib
import sys
import tempfile
import unittest

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import int8_lora_core as core  # noqa: E402


def make_lora(dtype=torch.bfloat16):
    g = torch.Generator().manual_seed(0)
    return {
        "layer.lora_up.weight": torch.randn(256, 16, generator=g).to(dtype),
        "layer.lora_down.weight": torch.randn(16, 128, generator=g).to(dtype),
        "layer.alpha": torch.tensor(16.0),
        "layer2.lokr_w1": torch.randn(8, 8, generator=g).to(dtype),
        "layer2.lokr_w2": torch.randn(128, 64, generator=g).to(dtype),
        "layer3.lora_down.weight": torch.randn(16, 8, 3, 3, generator=g).to(dtype),
        "layer3.dora_scale": torch.randn(32, 1, generator=g).to(dtype),
    }


def load_tensors(path):
    return core.load_int8_lora_with_metadata(path)[0]


def save_and_load(tensors, meta):
    """Write an INT8 file to a temp dir and load it back."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.safetensors")
        core.save_int8_lora(path, tensors, meta)
        return load_tensors(path)


def assert_report_matches_file(test, sd, tensors, meta, report):
    """The report's SSE must equal the SSE recomputed from the loaded file."""
    loaded = save_and_load(tensors, meta)
    sse = sum(
        float((sd[s.key].float() - loaded[s.key].float()).pow(2).sum(dtype=torch.float64))
        for s in report.stats
    )
    test.assertAlmostEqual(sse, report.total_sse, delta=report.total_sse * 1e-6)
    return loaded


class AbsmaxTests(unittest.TestCase):
    def test_roundtrip_error_is_small(self):
        w = torch.randn(64, 128)
        q, scale = core.quantize_absmax(w)
        self.assertEqual(q.dtype, torch.int8)
        self.assertEqual(scale.shape, (64,))
        back = core.dequantize_absmax(q, scale)
        # Max error per element is half a quantization step.
        step = scale[:, None]
        self.assertTrue(bool(((back - w).abs() <= step / 2 + 1e-6).all()))

    def test_tall_matrix_uses_column_scales(self):
        w = torch.randn(512, 16)
        q, scale = core.quantize_absmax(w)
        self.assertEqual(scale.shape, (16,))
        back = core.dequantize_absmax(q, scale)
        self.assertTrue(bool(((back - w).abs() <= scale[None, :] / 2 + 1e-6).all()))

    def test_scale_overhead_stays_small_for_low_rank(self):
        sd = {
            "a.lora_up.weight": torch.randn(4096, 16).bfloat16(),
            "a.lora_down.weight": torch.randn(16, 4096).bfloat16(),
        }
        _, _, report = core.quantize_state_dict(sd)
        self.assertLess(report.bytes_after / report.bytes_before, 0.52)

    def test_zero_row_stays_zero(self):
        w = torch.randn(4, 32)
        w[2] = 0
        q, scale = core.quantize_absmax(w)
        back = core.dequantize_absmax(q, scale)
        self.assertTrue(bool((back[2] == 0).all()))

    def test_conv_shape_preserved(self):
        w = torch.randn(8, 4, 3, 3)
        q, scale = core.quantize_absmax(w)
        self.assertEqual(q.shape, w.shape)
        self.assertEqual(core.dequantize_absmax(q, scale).shape, w.shape)

    def test_rejects_non_finite(self):
        bad = torch.randn(4, 4)
        bad[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            core.quantize_absmax(bad)


class StateDictTests(unittest.TestCase):
    def test_selection_and_report(self):
        sd = make_lora()
        tensors, meta, report = core.quantize_state_dict(sd, rotation=core.ROTATION_NONE)
        # Large factors are quantized, small/odd ones are not.
        self.assertEqual(tensors["layer.lora_up.weight"].dtype, torch.int8)
        self.assertIn("layer.lora_up.weight" + core.SCALE_SUFFIX, tensors)
        self.assertEqual(tensors["layer2.lokr_w1"].dtype, torch.bfloat16)  # 64 elems
        self.assertEqual(tensors["layer.alpha"].dtype, torch.float32)  # 0-D: kept
        self.assertEqual(tensors["layer3.dora_scale"].dtype, torch.bfloat16)
        self.assertEqual(meta[core.META_DTYPE], "bfloat16")
        self.assertEqual(meta[core.META_ROTATION], core.ROTATION_NONE)
        self.assertEqual(report.quantized_count, 4)
        self.assertGreater(report.total_sse, 0.0)
        self.assertLess(report.relative_sse, 1e-3)
        self.assertLess(report.bytes_after, report.bytes_before)
        self.assertIn("SSE", core.format_report(report, "a", "b"))

    def test_report_sse_matches_loaded_file(self):
        sd = make_lora()
        tensors, meta, report = core.quantize_state_dict(sd)
        assert_report_matches_file(self, sd, tensors, meta, report)

    def test_rejects_already_quantized_and_empty(self):
        with self.assertRaises(ValueError):
            core.quantize_state_dict({"a" + core.SCALE_SUFFIX: torch.ones(4)})
        with self.assertRaises(ValueError):
            core.quantize_state_dict({"layer.alpha": torch.tensor(1.0)})


class FileTests(unittest.TestCase):
    def roundtrip(self, dtype):
        sd = make_lora(dtype)
        tensors, meta, _ = core.quantize_state_dict(sd)
        loaded = save_and_load(tensors, meta)
        self.assertEqual(set(loaded), set(sd))
        for key, tensor in sd.items():
            self.assertEqual(loaded[key].shape, tensor.shape)
            self.assertEqual(loaded[key].dtype, tensor.dtype, key)
        return sd, loaded

    def test_roundtrip_bf16_and_fp16_and_fp32(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            sd, loaded = self.roundtrip(dtype)
            up = sd["layer.lora_up.weight"].float()
            err = (loaded["layer.lora_up.weight"].float() - up).pow(2).sum() / up.pow(2).sum()
            self.assertLess(float(err), 1e-3)
            # Untouched tensors are bit-identical.
            self.assertTrue(torch.equal(loaded["layer.alpha"], sd["layer.alpha"]))

    def test_rejects_unknown_files(self):
        with tempfile.TemporaryDirectory() as d:
            plain = os.path.join(d, "plain.safetensors")
            save_file({"a": torch.ones(2, 2)}, plain)
            with self.assertRaises(ValueError):
                load_tensors(plain)

            tensors, meta, _ = core.quantize_state_dict(make_lora())
            for key, value in (
                (core.META_FORMAT, "99"),
                (core.META_ROTATION, "spin"),
                (core.META_SCHEME, "groupwise"),
            ):
                path = os.path.join(d, f"{key}.safetensors")
                core.save_int8_lora(path, tensors, {**meta, key: value})
                with self.assertRaises(ValueError):
                    load_tensors(path)

    def test_rejects_missing_scale_and_orphan_scale(self):
        tensors, meta, _ = core.quantize_state_dict(make_lora())
        with tempfile.TemporaryDirectory() as d:
            missing = dict(tensors)
            del missing["layer.lora_up.weight" + core.SCALE_SUFFIX]
            p1 = os.path.join(d, "missing.safetensors")
            core.save_int8_lora(p1, missing, meta)
            with self.assertRaises(ValueError):
                load_tensors(p1)

            orphan = dict(tensors)
            del orphan["layer.lora_up.weight"]
            p2 = os.path.join(d, "orphan.safetensors")
            core.save_int8_lora(p2, orphan, meta)
            with self.assertRaises(ValueError):
                load_tensors(p2)


def outlier_lora(dtype=torch.bfloat16):
    g = torch.Generator().manual_seed(1)
    up = torch.randn(512, 16, generator=g)
    down = torch.randn(16, 1024, generator=g)
    # A few huge entries per row/column, the case rotation is meant to help.
    down[:, ::97] *= 40
    up[::37, :] *= 40
    return {
        "a.lora_up.weight": up.to(dtype),
        "a.lora_down.weight": down.to(dtype),
        "a.alpha": torch.tensor(8.0),
    }


def sign_lora():
    """Every value is +-1: unrotated int8 is exact, rotation can only hurt."""
    g = torch.Generator().manual_seed(2)
    sign = lambda *shape: (torch.randint(0, 2, shape, generator=g) * 2 - 1).float()
    return {
        "a.lora_up.weight": sign(512, 16),
        "a.lora_down.weight": sign(16, 1024),
    }


class ConvRotTests(unittest.TestCase):
    def test_hadamard_is_orthonormal_and_self_inverse(self):
        for group in core.CONVROT_GROUPS:
            h = core._hadamard(group)
            self.assertTrue(torch.allclose(h @ h, torch.eye(group), atol=1e-5))
            self.assertTrue(torch.equal(h, h.t()))

    def test_group_selection_depends_only_on_length(self):
        self.assertEqual(core._convrot_group_for(4096, 256), 256)
        self.assertEqual(core._convrot_group_for(3072, 256), 256)
        self.assertEqual(core._convrot_group_for(192, 256), 64)
        self.assertEqual(core._convrot_group_for(32, 256), 16)
        self.assertEqual(core._convrot_group_for(8, 256), 4)
        self.assertEqual(core._convrot_group_for(30, 256), 0)
        self.assertEqual(core._convrot_group_for(4096, 16), 16)

    def test_rotation_roundtrip_all_layouts(self):
        shapes = [(64, 128), (512, 16), (16, 8, 3, 3), (10, 105)]  # last: not rotatable
        for shape in shapes:
            w = torch.randn(*shape)
            q, scale = core.quantize_absmax(w, core.ROTATION_CONVROT)
            back = core.dequantize_absmax(q, scale, rotation=core.ROTATION_CONVROT)
            self.assertEqual(back.shape, w.shape)
            err = (back - w).pow(2).sum() / w.pow(2).sum()
            self.assertLess(float(err), 1e-3, shape)

    def test_non_divisible_axis_is_left_unrotated(self):
        self.assertFalse(core.is_rotated((10, 105), core.ROTATION_CONVROT, 256))
        self.assertTrue(core.is_rotated((512, 16), core.ROTATION_CONVROT, 256))
        self.assertFalse(core.is_rotated((512, 16), core.ROTATION_NONE, 256))
        w = torch.randn(10, 105)
        q0, s0 = core.quantize_absmax(w)
        q1, s1 = core.quantize_absmax(w, core.ROTATION_CONVROT)
        self.assertTrue(torch.equal(q0, q1) and torch.equal(s0, s1))

    def test_file_roundtrip_matches_report(self):
        sd = outlier_lora()
        tensors, meta, report = core.quantize_state_dict(sd, rotation=core.ROTATION_CONVROT)
        self.assertEqual(meta[core.META_ROTATION], core.ROTATION_CONVROT)
        self.assertEqual(meta[core.META_CONVROT_GROUP], "256")
        loaded = assert_report_matches_file(self, sd, tensors, meta, report)
        self.assertTrue(torch.equal(loaded["a.alpha"], sd["a.alpha"]))

    def test_rejects_bad_settings(self):
        sd = outlier_lora()
        with self.assertRaises(ValueError):
            core.quantize_state_dict(sd, rotation="spin")
        with self.assertRaises(ValueError):
            core.quantize_state_dict(sd, rotation=core.ROTATION_CONVROT, max_group=100)
        tensors, meta, _ = core.quantize_state_dict(sd, rotation=core.ROTATION_CONVROT)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad.safetensors")
            core.save_int8_lora(path, tensors, {**meta, core.META_CONVROT_GROUP: "100"})
            with self.assertRaises(ValueError):
                load_tensors(path)


class AutoRotationTests(unittest.TestCase):
    def test_auto_is_the_default(self):
        _, _, report = core.quantize_state_dict(outlier_lora())
        self.assertEqual(set(report.tried), {core.ROTATION_NONE, core.ROTATION_CONVROT})

    def test_auto_picks_convrot_on_outlier_data(self):
        sd = outlier_lora()
        _, meta, report = core.quantize_state_dict(sd, rotation=core.ROTATION_AUTO)
        self.assertEqual(report.rotation, core.ROTATION_CONVROT)
        self.assertEqual(meta[core.META_ROTATION], core.ROTATION_CONVROT)
        self.assertLess(report.tried[core.ROTATION_CONVROT], report.tried[core.ROTATION_NONE])
        self.assertEqual(report.total_sse, report.tried[core.ROTATION_CONVROT])
        self.assertIn("kept convrot", core.format_report(report))

    def test_auto_picks_none_when_rotation_hurts(self):
        sd = sign_lora()
        _, meta, report = core.quantize_state_dict(sd)
        self.assertEqual(report.rotation, core.ROTATION_NONE)
        self.assertEqual(meta[core.META_ROTATION], core.ROTATION_NONE)
        self.assertNotIn(core.META_CONVROT_GROUP, meta)
        self.assertLess(report.tried[core.ROTATION_NONE], report.tried[core.ROTATION_CONVROT])
        self.assertEqual(report.total_sse, report.tried[core.ROTATION_NONE])

    def test_auto_picks_none_when_nothing_can_rotate(self):
        g = torch.Generator().manual_seed(4)
        sd = {"a.lora_down.weight": torch.randn(10, 105, generator=g)}  # 105 is odd
        _, meta, report = core.quantize_state_dict(sd)
        self.assertEqual(report.rotation, core.ROTATION_NONE)
        self.assertEqual(report.rotated_count, 0)

    def test_forced_modes_do_not_try_both(self):
        for mode in core.ROTATIONS:
            _, meta, report = core.quantize_state_dict(outlier_lora(), rotation=mode)
            self.assertEqual(report.tried, {})
            self.assertEqual(meta[core.META_ROTATION], mode)

    def test_auto_file_loads_back(self):
        sd = outlier_lora()
        tensors, meta, report = core.quantize_state_dict(sd)
        assert_report_matches_file(self, sd, tensors, meta, report)


class MetadataTests(unittest.TestCase):
    SOURCE = {
        "ss_tag_frequency": '{"cat": 12}',
        "modelspec.title": "My style",
        "trigger_words": "mystyle",
        "int8_lora_format": "99",  # must not override ours
    }

    def roundtrip(self, source):
        sd = make_lora()
        tensors, meta, report = core.quantize_state_dict(
            sd, rotation=core.ROTATION_NONE, source_metadata=source
        )
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.safetensors")
            core.save_int8_lora(path, tensors, meta)
            state, back = core.load_int8_lora_with_metadata(path)
        return meta, back, report

    def test_source_metadata_is_carried_over_and_returned(self):
        meta, back, report = self.roundtrip(self.SOURCE)
        self.assertEqual(meta["ss_tag_frequency"], '{"cat": 12}')
        self.assertEqual(meta[core.META_FORMAT], core.FORMAT_VERSION)  # not "99"
        self.assertEqual(report.carried_metadata, 3)
        self.assertEqual(
            back,
            {
                "ss_tag_frequency": '{"cat": 12}',
                "modelspec.title": "My style",
                "trigger_words": "mystyle",
            },
        )
        self.assertIn("3 source keys carried over", core.format_report(report))

    def test_no_source_metadata(self):
        meta, back, report = self.roundtrip(None)
        self.assertEqual(back, {})
        self.assertEqual(report.carried_metadata, 0)
        self.assertIn("none in the source", core.format_report(report))

    def test_non_string_values_are_converted(self):
        meta, back, _ = self.roundtrip({"version": 5})
        self.assertEqual(back, {"version": "5"})

    def test_read_source_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            with_meta = os.path.join(d, "a.safetensors")
            save_file({"w": torch.ones(2, 2)}, with_meta, metadata={"ss_x": "1"})
            self.assertEqual(core.read_source_metadata(with_meta), {"ss_x": "1"})

            without = os.path.join(d, "b.safetensors")
            save_file({"w": torch.ones(2, 2)}, without)
            self.assertEqual(core.read_source_metadata(without), {})

            pickle_file = os.path.join(d, "c.pt")
            torch.save({"w": torch.ones(2, 2)}, pickle_file)
            self.assertEqual(core.read_source_metadata(pickle_file), {})
            self.assertEqual(core.read_source_metadata(os.path.join(d, "missing")), {})


class BatchTests(unittest.TestCase):
    NAMES = ["a.safetensors", "styles\\x.safetensors", "styles/deep\\y.safetensors", "other/z.pt"]

    def test_subfolders_include_every_depth_and_skip_the_root(self):
        self.assertEqual(core.lora_subfolders(self.NAMES), ["other", "styles", "styles/deep"])

    def test_select_in_folder_is_recursive_and_exact(self):
        self.assertEqual(
            core.select_in_folder(self.NAMES, "styles"),
            ["styles\\x.safetensors", "styles/deep\\y.safetensors"],
        )
        self.assertEqual(core.select_in_folder(self.NAMES, "styles/deep"), ["styles/deep\\y.safetensors"])
        self.assertEqual(core.select_in_folder(self.NAMES, "style"), [])  # no partial names

    def test_relative_output_path_keeps_the_subfolder(self):
        self.assertEqual(core.int8_relative_path("styles\\deep\\y.pt"), "styles/deep/y_int8.safetensors")
        self.assertEqual(core.int8_relative_path("a.safetensors"), "a_int8.safetensors")

    def test_candidate_check(self):
        core.check_batch_candidate(["x.lora_up.weight", "x.lora_down.weight"])
        core.check_batch_candidate(["x.lokr_w1", "x.lokr_w2"])
        core.check_batch_candidate(["x.diff", "x.diff_b"])  # full-weight diff LoRA
        for keys, reason in (
            (["model.diffusion_model.x.weight"], "look like a LoRA"),
            (["diffusion_model.blocks.0.attn.weight"], "look like a LoRA"),  # checkpoint
            (["x.lora_up.weight", "x.lora_up.weight" + core.SCALE_SUFFIX], "already an INT8"),
        ):
            with self.assertRaisesRegex(ValueError, reason):
                core.check_batch_candidate(keys)

    def test_run_batch_skips_failures_and_keeps_going(self):
        sd = make_lora()
        report = core.quantize_state_dict(sd)[2]

        def process(name):
            if name == "bad":
                raise ValueError("first line\nsecond line")
            if name == "exists":
                raise FileExistsError()
            return report

        ticks = []
        entries = core.run_batch(["ok1", "bad", "exists", "ok2"], process, lambda: ticks.append(1))
        self.assertEqual([bool(e.skipped) for e in entries], [False, True, True, False])
        self.assertEqual(entries[1].skipped, "first line")  # only the first line
        self.assertEqual(entries[2].skipped, "FileExistsError")  # falls back to the type
        self.assertEqual(len(ticks), 4)
        text = core.format_batch_report("styles", entries)
        self.assertIn("2 quantized, 2 skipped", text)
        self.assertIn("skip  bad | first line", text)
        self.assertIn("ok    ok1 -> ok1_int8.safetensors", text)
        self.assertIn("total:", text)

    def test_cancel_is_not_swallowed(self):
        class Cancel(BaseException):  # like ComfyUI's InterruptProcessingException
            pass

        def process(name):
            raise Cancel()

        with self.assertRaises(Cancel):
            core.run_batch(["a"], process)

    def test_end_to_end_on_real_files(self):
        with tempfile.TemporaryDirectory() as root:
            folder = os.path.join(root, "styles")
            os.makedirs(folder)
            good = os.path.join(folder, "good.safetensors")
            save_file(make_lora(), good, metadata={"ss_x": "1"})
            save_file({"model.layer.weight": torch.randn(64, 64)}, os.path.join(folder, "model.safetensors"))
            torch.save({"w": torch.ones(2, 2)}, os.path.join(folder, "old.pt"))
            tensors, meta, _ = core.quantize_state_dict(make_lora())
            core.save_int8_lora(os.path.join(folder, "already.safetensors"), tensors, meta)

            def process(name):
                path = os.path.join(root, *name.split("/"))
                keys = core.safetensors_keys(path)
                if keys is None:
                    raise ValueError("not a readable .safetensors file")
                core.check_batch_candidate(keys)
                _, _, report = core.quantize_state_dict(
                    load_file(path), source_metadata=core.read_source_metadata(path)
                )
                return report

            names = [f"styles/{n}" for n in ("already.safetensors", "good.safetensors", "model.safetensors", "old.pt")]
            entries = {e.name: e for e in core.run_batch(names, process)}

        self.assertEqual(entries["styles/good.safetensors"].skipped, "")
        self.assertEqual(entries["styles/good.safetensors"].report.carried_metadata, 1)
        self.assertEqual(entries["styles/already.safetensors"].skipped, "already an INT8 LoRA")
        self.assertEqual(entries["styles/model.safetensors"].skipped, "does not look like a LoRA")
        self.assertEqual(entries["styles/old.pt"].skipped, "not a readable .safetensors file")


class RestoreDtypeTests(unittest.TestCase):
    def quantized_file(self, dtype=torch.bfloat16):
        tensors, meta, _ = core.quantize_state_dict(make_lora(dtype), rotation=core.ROTATION_NONE)
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        path = os.path.join(d.name, "x.safetensors")
        core.save_int8_lora(path, tensors, meta)
        return path

    def load(self, path, dtype=None):
        return core.load_int8_lora_with_metadata(path, dtype)[0]

    def test_dropdown_values_map_to_dtypes(self):
        self.assertIsNone(core.restore_dtype("original"))
        self.assertEqual(core.restore_dtype("bf16"), torch.bfloat16)
        self.assertEqual(core.restore_dtype("fp16"), torch.float16)
        self.assertEqual(core.restore_dtype("fp32"), torch.float32)
        with self.assertRaises(ValueError):
            core.restore_dtype("fp8")

    def test_default_is_the_stored_dtype(self):
        path = self.quantized_file(torch.bfloat16)
        self.assertEqual(self.load(path)["layer.lora_up.weight"].dtype, torch.bfloat16)
        path = self.quantized_file(torch.float32)
        self.assertEqual(self.load(path)["layer.lora_up.weight"].dtype, torch.float32)

    def test_every_dtype_can_be_requested(self):
        path = self.quantized_file(torch.float32)
        for name, expected in (("bf16", torch.bfloat16), ("fp16", torch.float16), ("fp32", torch.float32)):
            loaded = self.load(path, core.restore_dtype(name))
            for key in ("layer.lora_up.weight", "layer.lora_down.weight", "layer2.lokr_w2"):
                self.assertEqual(loaded[key].dtype, expected, (name, key))

    def test_unquantized_tensors_keep_their_own_dtype(self):
        path = self.quantized_file(torch.bfloat16)
        loaded = self.load(path, torch.float16)
        self.assertEqual(loaded["layer.alpha"].dtype, torch.float32)  # stored as fp32
        self.assertEqual(loaded["layer3.dora_scale"].dtype, torch.bfloat16)  # not quantized
        self.assertEqual(loaded["layer2.lokr_w1"].dtype, torch.bfloat16)  # too small to quantize

    def test_lower_precision_is_just_a_cast_of_the_same_values(self):
        path = self.quantized_file(torch.float32)
        full = self.load(path, torch.float32)
        low = self.load(path, torch.bfloat16)
        for key in ("layer.lora_up.weight", "layer2.lokr_w2"):
            self.assertTrue(torch.equal(low[key], full[key].to(torch.bfloat16)), key)

    def test_bf16_halves_the_loaded_size_of_an_fp32_lora(self):
        path = self.quantized_file(torch.float32)

        def size(tensors):
            return sum(t.numel() * t.element_size() for k, t in tensors.items() if k.endswith(".weight") or "lokr" in k)

        self.assertAlmostEqual(size(self.load(path, torch.bfloat16)) / size(self.load(path)), 0.5, places=2)


class LoaderRobustnessTests(unittest.TestCase):
    """Damaged or unusual files, and how the stored dtype is chosen."""

    def save(self, tensors, meta):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        path = os.path.join(d.name, "x.safetensors")
        core.save_int8_lora(path, tensors, meta)
        return path

    def quantized(self, sd=None, rotation=core.ROTATION_NONE):
        return core.quantize_state_dict(sd or make_lora(), rotation=rotation)[:2]

    def test_float64_source_is_stored_and_restored_as_float32(self):
        tensors, meta = self.quantized(make_lora(torch.float64))
        self.assertEqual(meta[core.META_DTYPE], "float32")
        loaded = core.load_int8_lora_with_metadata(self.save(tensors, meta))[0]
        self.assertEqual(loaded["layer.lora_up.weight"].dtype, torch.float32)

    def test_mixed_dtypes_restore_to_the_most_common_one(self):
        sd = {
            "a.lora_up.weight": torch.randn(4096, 16).bfloat16(),
            "a.lora_down.weight": torch.randn(16, 4096).bfloat16(),
            "b.lora_up.weight": torch.randn(4096, 16).bfloat16(),
            "b.lora_down.weight": torch.randn(16, 4096),  # the odd one out, fp32
        }
        tensors, meta = self.quantized(sd)
        self.assertEqual(meta[core.META_DTYPE], "bfloat16")
        loaded = core.load_int8_lora_with_metadata(self.save(tensors, meta))[0]
        self.assertEqual({t.dtype for t in loaded.values()}, {torch.bfloat16})

    def test_dtype_override_works_with_rotation_too(self):
        tensors, meta = self.quantized(outlier_lora(), core.ROTATION_CONVROT)
        path = self.save(tensors, meta)
        full = core.load_int8_lora_with_metadata(path, torch.float32)[0]
        low = core.load_int8_lora_with_metadata(path, torch.bfloat16)[0]
        for key in ("a.lora_up.weight", "a.lora_down.weight"):
            self.assertTrue(torch.equal(low[key], full[key].to(torch.bfloat16)), key)

    def test_lower_precision_stays_close_to_the_original(self):
        sd = make_lora(torch.float32)
        tensors, meta = self.quantized(sd)
        low = core.load_int8_lora_with_metadata(self.save(tensors, meta), torch.bfloat16)[0]
        up = sd["layer.lora_up.weight"]
        err = (low["layer.lora_up.weight"].float() - up).pow(2).sum() / up.pow(2).sum()
        self.assertLess(float(err), 1e-3)

    def test_square_matrices_use_row_scales(self):
        # Pins the documented rule: only tall matrices switch to column scales.
        w = torch.randn(64, 64) * torch.linspace(0.1, 5.0, 64)[:, None]
        _, scale = core.quantize_absmax(w)
        self.assertTrue(torch.allclose(scale, w.abs().amax(dim=1) / 127.0))

    def test_scale_of_the_wrong_length_is_rejected(self):
        tensors, meta = self.quantized()
        tensors["layer.lora_up.weight" + core.SCALE_SUFFIX] = torch.ones(3)
        with self.assertRaisesRegex(ValueError, "scale has"):
            core.load_int8_lora_with_metadata(self.save(tensors, meta))

    def test_a_scaled_tensor_that_is_not_int8_is_rejected(self):
        tensors, meta = self.quantized()
        tensors["layer.lora_up.weight"] = tensors["layer.lora_up.weight"].float()
        with self.assertRaisesRegex(ValueError, "has a scale but is"):
            core.load_int8_lora_with_metadata(self.save(tensors, meta))

    def test_bad_metadata_values_are_rejected(self):
        tensors, meta = self.quantized(outlier_lora(), core.ROTATION_CONVROT)
        for key, value, message in (
            (core.META_CONVROT_GROUP, "abc", "invalid convrot group"),
            (core.META_CONVROT_GROUP, "100", "invalid convrot group"),
            (core.META_DTYPE, "float8", "unsupported stored dtype"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                core.load_int8_lora_with_metadata(self.save(tensors, {**meta, key: value}))


class CounterpartTests(unittest.TestCase):
    def test_mirrored_location_is_preferred_over_the_flat_one(self):
        names = ["a_int8.safetensors", "styles\\a_int8.safetensors"]
        self.assertEqual(core.find_int8_counterpart("styles/a.safetensors", names), "styles\\a_int8.safetensors")
        self.assertEqual(core.find_int8_counterpart("styles\\a.safetensors", names), "styles\\a_int8.safetensors")

    def test_falls_back_to_the_flat_file_name(self):
        self.assertEqual(
            core.find_int8_counterpart("styles/deep/b.safetensors", ["b_int8.safetensors"]), "b_int8.safetensors"
        )

    def test_root_loras_and_other_source_extensions(self):
        self.assertEqual(core.find_int8_counterpart("root.safetensors", ["root_int8.safetensors"]), "root_int8.safetensors")
        self.assertEqual(
            core.find_int8_counterpart("styles\\deep\\b.pt", ["styles/deep/b_int8.safetensors"]),
            "styles/deep/b_int8.safetensors",
        )

    def test_no_match_gives_none(self):
        self.assertIsNone(core.find_int8_counterpart("a.safetensors", []))
        self.assertIsNone(core.find_int8_counterpart("a.safetensors", ["other_int8.safetensors"]))
        self.assertIsNone(core.find_int8_counterpart("a.safetensors", ["xa_int8.safetensors"]))  # not a partial name
        self.assertIsNone(core.find_int8_counterpart("styles/a.safetensors", ["other/a_int8.safetensors"]))


class RunBatchCancelTests(unittest.TestCase):
    class Cancel(Exception):
        """Like ComfyUI's cancel exception in older versions: derives from Exception."""

    def test_listed_exceptions_stop_the_batch_even_though_they_are_exceptions(self):
        calls = []

        def process(name):
            calls.append(name)
            if name == "b":
                raise self.Cancel()
            return core.quantize_state_dict(make_lora())[2]

        with self.assertRaises(self.Cancel):
            core.run_batch(["a", "b", "c"], process, reraise=(self.Cancel,))
        self.assertEqual(calls, ["a", "b"])  # "c" never ran

    def test_without_reraise_an_exception_is_only_a_skipped_file(self):
        def process(name):
            if name == "b":
                raise self.Cancel()
            return core.quantize_state_dict(make_lora())[2]

        entries = core.run_batch(["a", "b", "c"], process)
        self.assertEqual([bool(e.skipped) for e in entries], [False, True, False])

    def test_other_errors_are_still_skipped_when_reraise_is_set(self):
        def process(name):
            if name == "bad":
                raise ValueError("not a LoRA")
            return core.quantize_state_dict(make_lora())[2]

        entries = core.run_batch(["a", "bad", "c"], process, reraise=(self.Cancel,))
        self.assertEqual([e.skipped for e in entries], ["", "not a LoRA", ""])


class StateSignatureTests(unittest.TestCase):
    def test_changes_when_files_appear_change_or_disappear(self):
        with tempfile.TemporaryDirectory() as d:
            one, two = os.path.join(d, "one.txt"), os.path.join(d, "two.txt")
            empty = core.state_signature([one, two])
            self.assertEqual(empty, core.state_signature([one, two]))  # stable
            pathlib.Path(one).write_text("a")
            appeared = core.state_signature([one, two])
            self.assertNotEqual(appeared, empty)
            pathlib.Path(one).write_text("longer content")  # different size
            edited = core.state_signature([one, two])
            self.assertNotEqual(edited, appeared)
            os.utime(one, ns=(1_000_000_000, 2_000_000_000))  # same size, different time
            self.assertNotEqual(core.state_signature([one, two]), edited)
            os.remove(one)
            self.assertEqual(core.state_signature([one, two]), empty)

    def test_depends_on_which_paths_are_listed(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = os.path.join(d, "a"), os.path.join(d, "b")
            self.assertNotEqual(core.state_signature([a]), core.state_signature([b]))
            self.assertEqual(core.state_signature([]), core.state_signature([]))


class NameTests(unittest.TestCase):
    def test_sanitize(self):
        self.assertEqual(core.sanitize_output_name("", "style"), "style_int8.safetensors")
        self.assertEqual(core.sanitize_output_name("My.safetensors", "x"), "My.safetensors")
        for bad in ("../evil", "a/b", "a\\b", "..", "c:x"):
            with self.assertRaises(ValueError):
                core.sanitize_output_name(bad, "x")


if __name__ == "__main__":
    unittest.main()
