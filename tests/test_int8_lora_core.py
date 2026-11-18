import os
import sys
import tempfile
import unittest

import torch
from safetensors.torch import save_file

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


def save_and_load(tensors, meta):
    """Write an INT8 file to a temp dir and load it back."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.safetensors")
        core.save_int8_lora(path, tensors, meta)
        return core.load_int8_lora(path)


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
                core.load_int8_lora(plain)

            tensors, meta, _ = core.quantize_state_dict(make_lora())
            for key, value in (
                (core.META_FORMAT, "99"),
                (core.META_ROTATION, "spin"),
                (core.META_SCHEME, "groupwise"),
            ):
                path = os.path.join(d, f"{key}.safetensors")
                core.save_int8_lora(path, tensors, {**meta, key: value})
                with self.assertRaises(ValueError):
                    core.load_int8_lora(path)

    def test_rejects_missing_scale_and_orphan_scale(self):
        tensors, meta, _ = core.quantize_state_dict(make_lora())
        with tempfile.TemporaryDirectory() as d:
            missing = dict(tensors)
            del missing["layer.lora_up.weight" + core.SCALE_SUFFIX]
            p1 = os.path.join(d, "missing.safetensors")
            core.save_int8_lora(p1, missing, meta)
            with self.assertRaises(ValueError):
                core.load_int8_lora(p1)

            orphan = dict(tensors)
            del orphan["layer.lora_up.weight"]
            p2 = os.path.join(d, "orphan.safetensors")
            core.save_int8_lora(p2, orphan, meta)
            with self.assertRaises(ValueError):
                core.load_int8_lora(p2)


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
                core.load_int8_lora(path)


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
            plain = core.load_int8_lora(path)
        self.assertEqual(set(state), set(plain))
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


class NameTests(unittest.TestCase):
    def test_sanitize(self):
        self.assertEqual(core.sanitize_output_name("", "style"), "style_int8.safetensors")
        self.assertEqual(core.sanitize_output_name("My.safetensors", "x"), "My.safetensors")
        for bad in ("../evil", "a/b", "a\\b", "..", "c:x"):
            with self.assertRaises(ValueError):
                core.sanitize_output_name(bad, "x")


if __name__ == "__main__":
    unittest.main()
