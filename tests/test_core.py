from __future__ import annotations

import argparse
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import torch
import yaml

from unicore_eeg import paths
from unicore_eeg import runtime as runtime_module
from unicore_eeg.config import (
    apply_overrides,
    deep_merge,
    explicit_overrides,
    get,
    load_config,
    resolve_settings,
    set_dotted,
    to_jsonable,
)
from unicore_eeg.ds004784 import task_to_labels
from unicore_eeg.manifest import code_version_hash, git_state, write_run_manifest
from unicore_eeg.model import ARTIFACT_NAMES, SparseRouter, UniCOREEGConfig, count_parameters
from unicore_eeg.physiomotion import map_annotation
from unicore_eeg.runtime import (
    TritonMissingError,
    check_ddp_allowed,
    configure_runtime,
    dataloader_kwargs,
)
from unicore_eeg.synthetic import PublicSignalPools, SyntheticEEGDataset


class RouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.router = SparseRouter(UniCOREEGConfig())

    @staticmethod
    def tokens(probabilities: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, experts = probabilities.shape
        return {
            "probabilities": probabilities,
            "severities": torch.ones(batch, experts),
            "priorities": torch.zeros(batch, experts),
        }

    def test_clean_bypass_has_zero_route(self) -> None:
        outputs = self.router(self.tokens(torch.full((2, 6), 0.1)))
        self.assertTrue(outputs["bypass"].all())
        self.assertEqual(float(outputs["route"].sum()), 0.0)

    def test_oracle_uses_only_labeled_experts(self) -> None:
        labels = torch.tensor([[1, 0, 1, 0, 0, 0]], dtype=torch.float32)
        outputs = self.router(self.tokens(torch.full((1, 6), 0.5)), mode="oracle", oracle_labels=labels)
        torch.testing.assert_close(outputs["route"], torch.tensor([[0.5, 0.0, 0.5, 0.0, 0.0, 0.0]]))

    def test_disabled_expert_is_never_routed(self) -> None:
        disabled = torch.tensor([[True, False, False, False, False, False]])
        outputs = self.router(self.tokens(torch.tensor([[0.99, 0.8, 0.7, 0.6, 0.5, 0.4]])), disabled_experts=disabled)
        self.assertEqual(float(outputs["route"][0, 0]), 0.0)

    def test_bypass_source_selects_presence_head_or_max_probability(self) -> None:
        tokens = self.tokens(torch.full((1, 6), 0.4))
        tokens["artifact_presence_probability"] = torch.tensor([0.9])
        presence = SparseRouter(UniCOREEGConfig(bypass_source="presence_head"))(tokens)
        maximum = SparseRouter(UniCOREEGConfig(bypass_source="max_probability"))(tokens)
        self.assertFalse(bool(presence["bypass"][0]))
        self.assertTrue(bool(maximum["bypass"][0]))


class DatasetTests(unittest.TestCase):
    def test_six_expert_record_shapes(self) -> None:
        sample = SyntheticEEGDataset(samples=1, length=256, sample_rate=128)[0]
        self.assertEqual(len(ARTIFACT_NAMES), 6)
        self.assertEqual(tuple(sample["artifacts"].shape), (6, 1, 256))
        self.assertEqual(tuple(sample["labels"].shape), (6,))

    def test_heldout_condition_targets_unknown(self) -> None:
        dataset = SyntheticEEGDataset(samples=12, length=256, sample_rate=128, mode="first_experiment")
        sample = dataset[7]
        self.assertTrue(sample["disabled_experts"][0])
        self.assertEqual(float(sample["label_mask"][0]), 0.0)
        self.assertEqual(float(sample["labels"][-1]), 1.0)

    def test_physiomotion_combination_mapping(self) -> None:
        labels, family = map_annotation("blink_hor_headm")
        self.assertEqual(float(labels[1]), 1.0)
        self.assertEqual(float(labels[4]), 1.0)
        self.assertEqual(family, 4)

    def test_ds004784_task_mapping(self) -> None:
        clean, clean_family = task_to_labels("Brain")
        all_labels, all_family = task_to_labels("All")
        self.assertEqual(float(clean.sum()), 0.0)
        self.assertEqual(clean_family, 0)
        self.assertEqual(float(all_labels[1]), 1.0)
        self.assertEqual(float(all_labels[2]), 1.0)
        self.assertEqual(float(all_labels[4]), 1.0)
        self.assertEqual(all_family, 5)


class PathSourceTests(unittest.TestCase):
    """T0.1：路径单一真相源。configs/_paths.yaml 必须只是 paths.py 的镜像。"""

    def test_paths_yaml_mirrors_paths_py(self) -> None:
        payload = yaml.safe_load((paths.CONFIG_ROOT / "_paths.yaml").read_text(encoding="utf-8"))
        self.assertEqual(Path(payload["project_root"]), paths.PROJECT_ROOT)
        self.assertEqual(Path(payload["data_root"]), paths.DATA_ROOT)
        self.assertEqual(Path(payload["raw_root"]), paths.RAW_ROOT)
        self.assertEqual(Path(payload["runs_root"]), paths.RUNS_ROOT)
        self.assertEqual(Path(payload["config_root"]), paths.CONFIG_ROOT)
        self.assertEqual(tuple(payload["pool_datasets"]), paths.POOL_DATASET_NAMES)
        self.assertEqual(paths.resolve_path(payload["external_pool"]), paths.EXTERNAL_POOL)
        self.assertEqual(paths.resolve_path(payload["pool_registry"]), paths.POOL_REGISTRY)

    def test_pool_covers_twelve_datasets(self) -> None:
        self.assertEqual(len(paths.POOL_DATASET_NAMES), 12)
        for name in paths.POOL_DATASET_NAMES:
            self.assertEqual(paths.pool_path(name), paths.POOL[name])

    def test_pool_paths_stay_under_external_pool(self) -> None:
        for name in paths.POOL_DATASET_NAMES:
            self.assertTrue(paths.is_in_pool(paths.pool_path(name, "probe")))

    def test_subject_subdirectory_does_not_override_dataset_root(self) -> None:
        # faced 在登记表里指向受试者子目录 nm000112，但数据集根必须仍是 faced/，
        # 否则 pool_path("faced", "feature_cache") 会拼到错的层级下。
        entry = paths.pool_entry("faced")
        self.assertFalse(entry.overridden)
        self.assertEqual(entry.root, paths.EXTERNAL_POOL / "faced")
        self.assertEqual(entry.declared, paths.EXTERNAL_POOL / "faced" / "nm000112")
        self.assertEqual(
            paths.pool_path("faced", "feature_cache"),
            paths.EXTERNAL_POOL / "faced" / "feature_cache",
        )

    def test_tuar_registered_as_unavailable(self) -> None:
        self.assertIn("declared_unavailable", paths.pool_roles("tuar"))

    def test_unknown_dataset_name_raises(self) -> None:
        with self.assertRaises(KeyError):
            paths.pool_path("does_not_exist")

    def test_assert_readonly_pool_blocks_pool_writes(self) -> None:
        with self.assertRaises(PermissionError):
            paths.assert_readonly_pool(paths.pool_path("bci2a", "x.gdf"))
        paths.assert_readonly_pool(paths.RAW_ROOT / "ok.txt")

    def test_pool_open_rejects_write_modes(self) -> None:
        with self.assertRaises(PermissionError):
            paths.pool_open(paths.pool_path("bci2a", "x.gdf"), "w")

    def test_resolve_path_symbols_and_relative_default(self) -> None:
        self.assertEqual(paths.resolve_path("@paths.RAW_ROOT"), paths.RAW_ROOT)
        self.assertEqual(paths.resolve_path("@paths.RUNS_ROOT/a/b"), paths.RUNS_ROOT / "a" / "b")
        self.assertEqual(paths.resolve_path("data/raw"), paths.RAW_ROOT)
        self.assertEqual(paths.resolve_path(paths.RAW_ROOT), paths.RAW_ROOT)
        with self.assertRaises(KeyError):
            paths.resolve_path("@paths.NOPE")


class ConfigTests(unittest.TestCase):
    """T0.4：配置基线与加载器。"""

    def test_base_config_loads_and_resolves_symbols(self) -> None:
        config = load_config("base.yaml")
        self.assertEqual(get(config, "signal.sample_rate"), 500)
        self.assertEqual(get(config, "signal.window_size"), 1000)
        self.assertEqual(get(config, "signal.bandpass.low"), 0.5)
        self.assertEqual(get(config, "signal.bandpass.high"), 45.0)
        self.assertEqual(get(config, "split.seeds"), [42, 1234, 20260922])
        self.assertEqual(get(config, "split.fractions.train"), 0.6)
        # §5.5 的硬约束必须写在基线里
        self.assertFalse(get(config, "runtime.torch_compile"))
        self.assertGreater(get(config, "dataloader.num_workers"), 0)
        self.assertFalse(get(config, "runtime.grad_scaler"))
        # @paths.* 必须被解析成绝对 Path，而不是留成字符串
        raw_root = get(config, "paths.raw_root")
        self.assertIsInstance(raw_root, Path)
        self.assertEqual(raw_root, paths.RAW_ROOT)

    def test_extends_merges_parent_then_child(self) -> None:
        config = load_config("multichannel_base.yaml")
        # 来自父配置
        self.assertEqual(get(config, "dataloader.batch_size"), 128)
        # 来自子配置
        self.assertEqual(get(config, "spatial.pca_rank"), 4)
        self.assertIn("montage", config)
        # 子配置不得把父配置的硬约束改坏
        self.assertFalse(get(config, "runtime.torch_compile"))

    def test_apply_overrides_does_not_mutate_source(self) -> None:
        # 回归测试：apply_overrides 曾用浅拷贝，嵌套字典被就地改写，
        # 导致覆盖一个实验会污染共用同一份 base 配置的其它实验。
        config = load_config("base.yaml")
        before = get(config, "dataloader.batch_size")
        overridden = apply_overrides(config, {"dataloader.batch_size": 64, "dataloader.num_workers": 0})
        self.assertEqual(get(overridden, "dataloader.batch_size"), 64)
        self.assertEqual(get(overridden, "dataloader.num_workers"), 0)
        self.assertEqual(get(config, "dataloader.batch_size"), before)
        self.assertNotEqual(get(config, "dataloader.num_workers"), 0)

    def test_deep_merge_is_recursive(self) -> None:
        merged = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 9, "d": 3}})
        self.assertEqual(merged, {"a": {"b": 1, "c": 9, "d": 3}})

    def test_get_and_set_dotted(self) -> None:
        config: dict = {}
        set_dotted(config, "a.b.c", 7)
        self.assertEqual(get(config, "a.b.c"), 7)
        self.assertIsNone(get(config, "a.b.missing"))
        self.assertEqual(get(config, "a.b.missing", "fallback"), "fallback")

    def test_explicit_overrides_ignores_unset(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--epochs", type=int, default=argparse.SUPPRESS)
        parser.add_argument("--lr", type=float, default=argparse.SUPPRESS)
        self.assertEqual(explicit_overrides(parser.parse_args([])), {})
        self.assertEqual(explicit_overrides(parser.parse_args(["--epochs", "3"])), {"epochs": 3})

    def test_to_jsonable_converts_paths(self) -> None:
        payload = to_jsonable({"root": paths.RAW_ROOT, "items": [paths.RUNS_ROOT]})
        self.assertEqual(payload["root"], str(paths.RAW_ROOT))
        self.assertEqual(payload["items"], [str(paths.RUNS_ROOT)])
        json.dumps(payload)  # 必须可序列化

    def test_circular_extends_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "a.yaml"
            second = Path(directory) / "b.yaml"
            first.write_text("extends: b.yaml\n", encoding="utf-8")
            second.write_text("extends: a.yaml\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(first)

    def test_missing_config_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_config("definitely_not_a_config.yaml")


class ProbeHelperTests(unittest.TestCase):
    """T0.3：scripts/probe_pool.py 的纯函数（脚本不是包，按路径加载）。"""

    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location(
            "probe_pool_under_test", paths.PROJECT_ROOT / "scripts" / "probe_pool.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.probe = module

    def test_describe_array_handles_mapping(self) -> None:
        # 回归测试：曾用 isinstance(x, np.ndarray.dtype)，而 NumPy 2.x 里它是
        # 描述符不是类型，遇到 mat 的 struct（dict-like）就会抛 TypeError。
        info = self.probe.describe_array({"a": 1, "b": 2})
        self.assertEqual(info["kind"], "mapping")
        self.assertEqual(info["n_keys"], 2)

    def test_describe_array_handles_arbitrary_object(self) -> None:
        info = self.probe.describe_array(object())
        self.assertEqual(info["kind"], "object")

    def test_describe_array_zero_dim_carries_value(self) -> None:
        info = self.probe.describe_array(np.array(250.0))
        self.assertEqual(info["kind"], "ndarray")
        self.assertEqual(info["value"], 250.0)

    def test_describe_array_channel_vector_carries_values(self) -> None:
        info = self.probe.describe_array(np.array(["FC5", "C3", "CP6"]))
        self.assertEqual(info["values"], ["FC5", "C3", "CP6"])

    def test_describe_array_large_matrix_reports_shape_only(self) -> None:
        info = self.probe.describe_array(np.zeros((4, 5), dtype=np.float32))
        self.assertEqual(info["shape"], [4, 5])
        self.assertNotIn("values", info)

    def test_human_bytes(self) -> None:
        self.assertEqual(self.probe.human_bytes(0), "0.00 B")
        self.assertEqual(self.probe.human_bytes(1024), "1.00 KiB")
        self.assertEqual(self.probe.human_bytes(2 * 1024**3), "2.00 GiB")

    def test_walk_files_skips_pycache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "keep.txt").write_text("x", encoding="utf-8")
            cache = root / "__pycache__"
            cache.mkdir()
            (cache / "skip.pyc").write_bytes(b"x")
            names = sorted(item.name for item in self.probe.walk_files(root))
            self.assertEqual(names, ["keep.txt"])


class ResolveSettingsTests(unittest.TestCase):
    """T0.7：命令行 > 配置文件 > 代码默认值。"""

    def test_cli_beats_config_beats_defaults(self) -> None:
        defaults = {"batch_size": 24, "num_workers": 8, "epochs": 4, "out": Path("x")}
        mapping = {"batch_size": "dataloader.batch_size", "num_workers": "dataloader.num_workers"}

        # 没有 --config：全部用代码默认值
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=Path, default=None)
        parser.add_argument("--batch-size", type=int, default=argparse.SUPPRESS)
        settings, config = resolve_settings(parser.parse_args([]), defaults, None, mapping)
        self.assertEqual(settings["batch_size"], 24)
        self.assertIsNone(config)

        # 有 --config：配置覆盖默认值（base.yaml 的 batch_size=128、num_workers=8）
        settings, config = resolve_settings(
            parser.parse_args([]), defaults, "base.yaml", mapping
        )
        self.assertEqual(settings["batch_size"], 128)
        self.assertIsNotNone(config)
        # 配置里没有的项保持默认值
        self.assertEqual(settings["epochs"], 4)

        # 命令行显式给出：优先级最高
        settings, _ = resolve_settings(
            parser.parse_args(["--batch-size", "32"]), defaults, "base.yaml", mapping
        )
        self.assertEqual(settings["batch_size"], 32)

    def test_suppress_keeps_unset_flags_out_of_namespace(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--flag", action="store_true", default=argparse.SUPPRESS)
        self.assertNotIn("flag", vars(parser.parse_args([])))
        self.assertTrue(parser.parse_args(["--flag"]).flag)


class RuntimeTests(unittest.TestCase):
    """T0.7 / §5.5：运行配置的硬约束必须是可执行检查。"""

    def test_dataloader_kwargs_adds_persistent_workers_only_when_needed(self) -> None:
        device = torch.device("cpu")
        zero = dataloader_kwargs(64, 0, device)
        self.assertNotIn("persistent_workers", zero)
        self.assertNotIn("prefetch_factor", zero)
        many = dataloader_kwargs(64, 8, device)
        self.assertTrue(many["persistent_workers"])
        self.assertEqual(many["prefetch_factor"], 4)

    def test_dataloader_kwargs_rejects_batch_above_safe_limit(self) -> None:
        with self.assertRaises(ValueError):
            dataloader_kwargs(runtime_module.MAX_BATCH_SIZE + 1, 8, torch.device("cpu"))
        with self.assertRaises(ValueError):
            dataloader_kwargs(128, -1, torch.device("cpu"))

    def test_configure_runtime_defaults(self) -> None:
        settings = configure_runtime(None, device=torch.device("cpu"))
        self.assertEqual(settings["precision"], "bf16")
        self.assertFalse(settings["grad_scaler"])
        self.assertFalse(settings["torch_compile"])

    def test_configure_runtime_rejects_compile_without_triton(self) -> None:
        try:
            import triton  # noqa: F401
        except ImportError:
            pass
        else:
            self.skipTest("triton is installed on this machine")
        with self.assertRaises(TritonMissingError):
            configure_runtime({"runtime": {"torch_compile": True}})

    def test_configure_runtime_reads_base_yaml(self) -> None:
        base = load_config("base.yaml")
        settings = configure_runtime(base, device=torch.device("cpu"))
        # base.yaml 里 torch_compile 必须是 false（§5.5 硬约束）
        self.assertFalse(settings["torch_compile"])
        self.assertEqual(settings["matmul_precision"], "high")

    def test_check_ddp_allowed_always_rejects(self) -> None:
        with self.assertRaises(RuntimeError):
            check_ddp_allowed(2, True)
        check_ddp_allowed(2, False)


class ManifestTests(unittest.TestCase):
    """T0.7 / §5.3：run 目录登记。"""

    def test_code_version_hash_is_stable(self) -> None:
        first = code_version_hash()
        second = code_version_hash()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        int(first, 16)  # 必须是合法十六进制

    def test_write_run_manifest_creates_manifest_and_config_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = load_config("base.yaml")
            target = Path(directory) / "run"
            path = write_run_manifest(
                target,
                command=["python", "scripts/first_experiment.py"],
                settings={"seed": 42, "out": target},
                config=config,
                dataset={"name": "synthetic", "channels": 1, "sample_rate": 500},
                extra={"route_modes": ["oracle", "learned", "all"]},
            )
            self.assertTrue(path.exists())
            self.assertTrue((target / "config.yaml").exists())
            payload = json.loads(path.read_text(encoding="utf-8"))
            for key in (
                "git",
                "code_version_hash",
                "command",
                "settings",
                "config",
                "config_file",
                "dataset",
                "extra",
                "written_utc",
            ):
                self.assertIn(key, payload)
            self.assertEqual(payload["dataset"]["sample_rate"], 500)
            # Path 必须转成字符串才能序列化
            self.assertIsInstance(payload["settings"]["out"], str)

    def test_write_run_manifest_refuses_pool_target(self) -> None:
        with self.assertRaises(PermissionError):
            write_run_manifest(paths.EXTERNAL_POOL / "should_not_exist")

    def test_git_state_reports_head(self) -> None:
        state = git_state()
        if not state["available"]:
            self.skipTest(f"not a git repository: {state.get('detail')}")
        self.assertEqual(len(state["head"]), 40)
        self.assertIn("dirty", state)


class MontageTests(unittest.TestCase):
    """T1.1 验收：montage 与坐标表。"""

    def test_every_montage_resolves_fully(self) -> None:
        """每个 montage 配置：解析通道数与配置一致，mask 全 True。"""
        from unicore_eeg import montage as M

        names = M.list_montages()
        self.assertGreaterEqual(len(names), 12, f"montage 数量偏少：{names}")
        for name in names:
            with self.subTest(montage=name):
                spec = M.load_montage(name)
                coords, mask = M.resolve_montage(spec.channels, name)
                self.assertEqual(coords.shape, (spec.channel_count, 3))
                self.assertEqual(mask.shape, (spec.channel_count,))
                self.assertTrue(bool(mask.all()), f"{name} 有未解析通道")
                self.assertFalse(torch.isnan(coords).any())
                radii = coords.norm(dim=-1)
                if spec.bipolar:
                    # 双极导联取中点；跨半球导联（C4-A1、FT9-FT10）的中点会靠近头心，
                    # 下界只要求严格为正，具体低信息位由 test_low_information_positions 单独盯。
                    self.assertTrue(bool((radii > 0.1).all()), f"{name} 有坐标退化到原点")
                else:
                    self.assertTrue(
                        bool((radii > 0.8).all()),
                        f"{name} 的单极坐标应落在头部表面附近，实际最小半径 {float(radii.min()):.3f}",
                    )

    def test_unknown_channel_is_masked_without_raising(self) -> None:
        """不存在的通道名：mask 置 False、坐标置零、不抛异常（手册 T1.1 验收第 2 条）。"""
        from unicore_eeg import montage as M

        coords, mask = M.resolve_montage(["Fp1", "NOT_AN_ELECTRODE", "Cz"])
        self.assertEqual(mask.tolist(), [True, False, True])
        self.assertTrue(torch.allclose(coords[1], torch.zeros(3)))
        self.assertEqual(len(M.classify_channel("NOT_AN_ELECTRODE").kind), len("unresolved"))

    def test_bipolar_coordinate_is_midpoint(self) -> None:
        """双极导联坐标 = 两端电极坐标的中点（手册 T1.1 步骤 5）。"""
        from unicore_eeg import montage as M

        spec = M.load_montage("physiomotion")
        index = spec.index_of("Fp1-F7")
        self.assertIsNotNone(index)
        first = M.resolve_electrode("Fp1")
        second = M.resolve_electrode("F7")
        expected = (first[1] + second[1]) / 2.0
        self.assertTrue(torch.allclose(spec.coords[index], expected, atol=1e-6))
        self.assertEqual(spec.links[index], ("Fp1", "F7"))
        self.assertTrue(spec.bipolar)
        self.assertEqual(M.coverage_report("physiomotion")["kinds"], {"bipolar": 34})

    def test_name_variants_all_canonicalize(self) -> None:
        """实测里出现的各种写法都要解析到同一个电极。"""
        from unicore_eeg import montage as M

        cases = {
            "EEG:C3": "C3",
            "EEG-Cz": "Cz",
            "C3..": "C3",
            "FCz_ref": "FCz",
            "Oz_ref": "Oz",
            "EEG Fp1-Ref": "Fp1",
            "FP1": "Fp1",
            "IZ": "Iz",
            "PO10": "PO10",
            "FTT9h": "FTT9h",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                found = M.resolve_electrode(raw)
                self.assertIsNotNone(found, f"{raw} 没解析出来")
                self.assertEqual(found[0], expected)
        # T3/T4/T5/T6 与 T7/T8/P7/P8 在标准表里是重合位置
        for old, new in (("T3", "T7"), ("T4", "T8"), ("T5", "P7"), ("T6", "P8")):
            with self.subTest(pair=f"{old}/{new}"):
                self.assertTrue(
                    torch.allclose(M.resolve_electrode(old)[1], M.resolve_electrode(new)[1])
                )

    def test_positional_mapping_uses_recorded_layout(self) -> None:
        """bci2a 的 EDF 通道名是匿名的，只能按位置映射；3 路 EOG 不参与空间坐标。"""
        from unicore_eeg import montage as M

        spec = M.load_montage("bci2a")
        self.assertEqual(spec.resolve, "positional")
        raw = list(spec.meta["raw_layout"])
        self.assertEqual(len(raw), 25)

        coords, mask = M.resolve_montage(raw[:22], "bci2a")
        self.assertTrue(bool(mask.all()))
        self.assertEqual(coords.shape, (22, 3))
        # 锚点位置必须落到正确的标准名
        for index, expected in ((0, "Fz"), (7, "C3"), (9, "Cz"), (11, "C4"), (19, "Pz")):
            self.assertEqual(M.classify_channel(raw[index], spec.coordinates, spec.raw_map).standard, expected)
        # EOG 通道没有空间坐标，应该被 mask 掉而不是硬塞一个位置
        eog = M.resolve_montage(raw[22:], "bci2a")
        self.assertFalse(bool(eog[1].any()))

    def test_positional_anchor_mismatch_raises(self) -> None:
        """锚点校验失败必须报错，而不是默默按错位映射。"""
        from unittest import mock

        from unicore_eeg import montage as M

        payload = yaml.safe_load((M.MONTAGE_DIR / "bci2a.yaml").read_text(encoding="utf-8"))
        payload["raw_layout"] = list(reversed(payload["raw_layout"]))
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "broken.yaml").write_text(
                yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8"
            )
            with mock.patch.object(M, "MONTAGE_DIR", directory):
                M.load_montage.cache_clear()
                try:
                    with self.assertRaises(M.MontageError):
                        M.load_montage("broken")
                finally:
                    M.load_montage.cache_clear()

    def test_channel_count_mismatch_raises(self) -> None:
        """channel_count 与 channels 长度矛盾时必须报错（防止配置悄悄漂移）。"""
        from unittest import mock

        from unicore_eeg import montage as M

        payload = yaml.safe_load((M.MONTAGE_DIR / "bci2b.yaml").read_text(encoding="utf-8"))
        payload["channel_count"] = 99
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "bad_count.yaml").write_text(
                yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8"
            )
            with mock.patch.object(M, "MONTAGE_DIR", directory):
                M.load_montage.cache_clear()
                try:
                    with self.assertRaises(M.MontageError):
                        M.load_montage("bad_count")
                finally:
                    M.load_montage.cache_clear()

    def test_bci2b_is_not_bipolar(self) -> None:
        """bci2b 的 3 路没有第二电极，按标准位取坐标（与手册字面写法不同的处置）。"""
        from unicore_eeg import montage as M

        spec = M.load_montage("bci2b")
        self.assertFalse(spec.bipolar)
        self.assertEqual(M.coverage_report("bci2b")["kinds"], {"eeg": 3})

    def test_low_information_positions_are_reported(self) -> None:
        """跨半球双极导联的中点靠近头心，必须被如实标出而不是伪装成正常位置。"""
        from unicore_eeg import montage as M

        detected = {
            name: [item[0] for item in M.coverage_report(name)["low_information"]]
            for name in M.list_montages()
        }
        self.assertEqual(detected["cap_sleep"], ["C4-A1"])
        self.assertEqual(detected["chbmit"], ["FT9-FT10"])
        flat = [channel for channels in detected.values() for channel in channels]
        self.assertEqual(sorted(flat), ["C4-A1", "FT9-FT10"])

        spec = M.load_montage("cap_sleep")
        index = spec.index_of("C4-A1")
        self.assertLess(float(spec.coords[index].norm()), 0.6)
        self.assertEqual(spec.links[index], ("C4", "A1"))

    def test_reference_table_reproduces_mne(self) -> None:
        """参考坐标表必须与 MNE 的 standard_1020 数值一致（可复核，不手抄）。"""
        try:
            import mne
        except ImportError:  # pragma: no cover
            self.skipTest("mne 未安装")
        import warnings

        from unicore_eeg import montage as M

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            positions = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
        normals = np.linalg.norm(np.array([positions[name] for name in M.REFERENCE_19_NAMES]), axis=1)
        scale = float(normals.mean())

        for name in list(positions)[:40] + ["Fpz", "Oz", "Iz", "A1", "A2", "T3", "T7"]:
            with self.subTest(name=name):
                found = M.resolve_electrode(name)
                self.assertIsNotNone(found)
                raw = np.array(positions[name], dtype=np.float64)
                self.assertTrue(
                    np.allclose(found[1].numpy(), raw / scale, atol=1e-6),
                    f"{name} 与 MNE 不一致：{found[1].numpy()} vs {raw / scale}",
                )

        # 归一化必须让标准 19 导的平均半径为 1
        radii = [float(M.resolve_electrode(name)[1].norm()) for name in M.REFERENCE_19_NAMES]
        self.assertAlmostEqual(float(np.mean(radii)), 1.0, places=5)

    def test_montages_convert_to_common_head_axes(self) -> None:
        """标准坐标与 ds004784 坐标都按 right/anterior/superior 输出。"""
        from unicore_eeg import montage as M

        reference = M.load_montage("standard_reference")
        phantom = M.load_montage("ds004784")
        fp1_head = reference.to_head_ras(torch.tensor([reference.coordinates["Fp1"]]))[0]
        a1_head = phantom.to_head_ras(torch.tensor(phantom.coordinates["A1"]))
        self.assertGreater(float(fp1_head[1]), 0.5)
        self.assertGreater(float(a1_head[2]), 0.99)
        self.assertTrue(torch.allclose(a1_head[:2], torch.zeros(2), atol=1e-6))

    def test_project_to_disk(self) -> None:
        """二维投影：有效通道落在单位圆内，被 mask 的通道严格为零。"""
        from unicore_eeg import montage as M

        spec = M.load_montage("physiomotion")
        projected = M.project_to_disk(spec.coords, spec.mask)
        self.assertEqual(projected.shape, (spec.channel_count, 2))
        self.assertLessEqual(float(projected.norm(dim=-1).max()), 1.0 + 1e-6)
        # 顶点（极角最小）应在圆盘中心附近，枕区在边缘
        polar = torch.acos(spec.coords[:, 2].clamp(-1, 1))
        self.assertTrue(float(projected[polar.argmin()].norm()) < float(projected[polar.argmax()].norm()))

        masked = M.project_to_disk(
            torch.cat((spec.coords, torch.zeros(2, 3))),
            torch.cat((spec.mask, torch.zeros(2, dtype=torch.bool))),
        )
        self.assertTrue(torch.allclose(masked[-2:], torch.zeros(2, 2)))

    def test_ds004784_geometry_regression(self) -> None:
        """把 T1.1 实测到的体模几何事实固化成回归断言。"""
        from unicore_eeg import montage as M

        spec = M.load_montage("ds004784")
        self.assertEqual(spec.channel_count, 128)
        radii = spec.coords.norm(dim=-1)
        self.assertLess(float((radii - 1.0).abs().max()), 1e-6, "归一化后应精确落在单位球面")

        # 命名是 ABCD 四扇区各 32 个，不是 A1..A128
        expected = [f"{sector}{index}" for sector in "ABCD" for index in range(1, 33)]
        self.assertEqual(list(spec.channels), expected)

        # 严格左右镜像对称（y → -y）——这条事实使"面内朝向"不可解
        points = {tuple(round(float(v), 6) for v in row) for row in spec.coords}
        mirrored = {(x, -y, z) for x, y, z in points}
        self.assertEqual(points, mirrored, "体模电极阵不是左右镜像对称，与实测结论矛盾")

    def test_ds004784_has_no_19ch_subset(self) -> None:
        """经用户确认放弃 19 导子集：球面阵无法可靠映射到标准 10-20。"""
        from unicore_eeg import montage as M

        self.assertFalse((M.MONTAGE_DIR / "ds004784_19ch.yaml").exists())
        with self.assertRaises(M.MontageError):
            M.load_montage("ds004784_19ch")


class ObservationExtractorTests(unittest.TestCase):
    """T1.2 验收：观测抽取器多通道化。"""

    def test_output_shapes_are_per_channel(self) -> None:
        """C=1 与 C=4 均前向成功，6 个输出 shape 均为 (B, C, T)。"""
        from unicore_eeg.model import ArtifactObservationExtractor

        extractor = ArtifactObservationExtractor(sample_rate=500)
        for channels in (1, 4):
            with self.subTest(channels=channels):
                y = torch.randn(3, channels, 1000)
                observations = extractor(y)
                self.assertEqual(len(observations), len(ARTIFACT_NAMES))
                for index, item in enumerate(observations):
                    self.assertEqual(
                        tuple(item.shape),
                        (3, channels, 1000),
                        f"第 {index} 项观测（{ARTIFACT_NAMES[index]}）形状不对",
                    )
                    self.assertFalse(torch.isnan(item).any())

    def test_channels_are_no_longer_identical(self) -> None:
        """C=4 时各通道观测互不相同（改造前谐波基被 expand_as 复制到全通道）。"""
        from unicore_eeg.model import ArtifactObservationExtractor

        torch.manual_seed(0)
        y = torch.randn(2, 4, 1000)
        observations = ArtifactObservationExtractor(sample_rate=500)(y)
        for index, name in enumerate(ARTIFACT_NAMES):
            with self.subTest(observation=name):
                self.assertFalse(
                    torch.allclose(observations[index][:, 0], observations[index][:, 1]),
                    f"{name} 的通道 0 与通道 1 完全相同，说明仍在共享同一份观测",
                )

    def test_cardiac_lags_are_per_channel(self) -> None:
        """左通道 0.8 s 周期、右通道 1.2 s 周期 → 两者滞后必须不同。"""
        from unicore_eeg.model import ArtifactObservationExtractor

        sample_rate = 500
        length = 3000
        time = torch.arange(length).float() / sample_rate
        left = torch.sin(2 * np.pi * time / 0.8)
        right = torch.sin(2 * np.pi * time / 1.2)
        y = torch.stack((left, right))[None]  # (1, 2, T)

        extractor = ArtifactObservationExtractor(sample_rate=sample_rate)
        lags = extractor.cardiac_lags(y)
        self.assertEqual(tuple(lags.shape), (1, 2))
        self.assertNotEqual(int(lags[0, 0]), int(lags[0, 1]))
        self.assertAlmostEqual(int(lags[0, 0]), int(0.8 * sample_rate), delta=15)
        self.assertAlmostEqual(int(lags[0, 1]), int(1.2 * sample_rate), delta=20)

        # 每个通道的脉冲必须按自己的滞后构造
        pulse = extractor._cardiac(y, lags)
        self.assertEqual(tuple(pulse.shape), (1, 2, length))
        self.assertFalse(torch.allclose(pulse[:, 0], pulse[:, 1]))

    def test_harmonic_fit_equals_direct_lstsq(self) -> None:
        """C=1 时新的正规方程解必须与直接 lstsq 一致（防止改造悄悄改变数值行为）。"""
        from unicore_eeg.model import ArtifactObservationExtractor

        torch.manual_seed(3)
        y = torch.randn(2, 1, 1000)
        extractor = ArtifactObservationExtractor(sample_rate=500)
        fitted = extractor._harmonic_basis(y)

        peak = extractor.peak_frequency(y)
        time = torch.arange(y.size(-1)).float() / 500.0
        basis = []
        for harmonic in (1.0, 2.0, 3.0):
            phase = 2 * np.pi * harmonic * peak[:, None] * time[None]
            basis.extend((torch.sin(phase), torch.cos(phase)))
        design = torch.stack(basis, dim=-1)  # (B, T, 6)

        reference = torch.empty_like(y)
        for batch in range(y.size(0)):
            solution = torch.linalg.lstsq(design[batch], y[batch, 0, :, None]).solution
            reference[batch, 0] = (design[batch] @ solution).squeeze(-1)
        self.assertTrue(torch.allclose(fitted, reference, atol=1e-3))

    def test_full_model_forward_multichannel(self) -> None:
        """整模型在 C=1 与 C=4 下都能前向（观测直接喂给专家的 Conv1d）。"""
        from unicore_eeg.model import UniCOREEG

        for channels in (1, 4):
            with self.subTest(channels=channels):
                model = UniCOREEG(UniCOREEGConfig(in_channels=channels, sample_rate=500))
                outputs = model(torch.randn(2, channels, 1000))
                self.assertEqual(tuple(outputs["clean"].shape), (2, channels, 1000))
                self.assertEqual(
                    tuple(outputs["artifact_components_norm"].shape),
                    (2, len(ARTIFACT_NAMES), channels, 1000),
                )
                self.assertFalse(torch.isnan(outputs["clean"]).any())


    def test_extractor_works_under_autocast(self) -> None:
        """autocast 打开时必须能算通（用 CPU 的 bf16 autocast 复现 CUDA 上的场景）。

        训练时 ``robust_normalize`` 的除法会被 autocast 推成 bf16，观测抽取器因此拿到
        bf16 张量。但即使把张量转成 float32 也不够：``@`` 本身是 autocast 的降精度算子，
        会把 float32 的输入再降回 bf16，于是 ``torch.linalg.solve`` 报
        ``lu_factor_cublas / lu_cpu not implemented for 'BFloat16'``。
        正确做法是让抽取器显式退出 autocast（``model.no_autocast``）。
        """
        from unicore_eeg.model import ArtifactObservationExtractor

        extractor = ArtifactObservationExtractor(sample_rate=500)
        y = torch.randn(2, 3, 1000)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            observations = extractor(y)
        self.assertEqual(len(observations), len(ARTIFACT_NAMES))
        for item in observations:
            self.assertFalse(torch.isnan(item.float()).any())

        # 直接喂 bf16 张量（不带 autocast）也必须能算：FFT 在 CPU 上不支持 bf16，
        # 抽取器内部已经统一转成 float32 计算，输出再转回输入 dtype。
        bf16 = y.to(torch.bfloat16)
        converted = extractor(bf16)
        self.assertEqual(converted[0].dtype, torch.bfloat16)
        self.assertFalse(torch.isnan(converted[0].float()).any())


class SpatialMixerTests(unittest.TestCase):
    """T1.4 验收：合成器真混音。"""

    #: 一个 8 导前/后对照布局（前额四导 + 枕区两导 + 中央两导）
    LAYOUT = ("Fp1", "Fp2", "F7", "F8", "O1", "O2", "C3", "C4")

    def setUp(self) -> None:
        from unicore_eeg import montage as M
        from unicore_eeg.spatial_mixer import SpatialMixer

        self.montage = M
        coords, mask = M.resolve_montage(self.LAYOUT)
        self.coords, self.mask = coords, mask
        self.mixer = SpatialMixer(coords, sample_rate=500, mask=mask)
        self.generator = torch.Generator().manual_seed(20260922)

    def _weights(self, kind: str) -> list[float]:
        return self.mixer.build_mixing_matrix(kind, torch.Generator().manual_seed(7)).tolist()

    def test_frame_axes_controls_ocular_direction(self) -> None:
        """前额优势必须按 montage 声明的 `anterior` 轴算，而不是写死 +y。

        ds004784 的体模坐标帧是 x=前后（anterior = (1,0,0)），不是 MNE 的 RAS。
        若忽略 `frame_axes`，该数据集上的眼动先验会指向左右而不是前后——
        权重与真实前后轴的相关会掉到 0 附近。这条测试用"权重随 anterior 单调上升"
        来钉住方向正确，两个坐标系各验一次。
        """
        from unicore_eeg.spatial_mixer import SpatialMixer

        for name in ("standard_reference", "ds004784"):
            with self.subTest(montage=name):
                spec = self.montage.load_montage(name)
                coords, mask = spec.coords, spec.mask
                mixer = SpatialMixer(coords, sample_rate=500, mask=mask, frame_axes=spec.frame_axes)
                weights = mixer.build_mixing_matrix("ocular_drift", torch.Generator().manual_seed(0))
                anterior = coords @ mixer.axes["anterior"]
                correlation = float(torch.corrcoef(torch.stack((anterior, weights)))[0, 1])
                extreme = float(weights[anterior.argmax()] / weights[anterior.argmin()])
                self.assertGreater(correlation, 0.8, f"{name}: 权重与 anterior 轴的相关仅 {correlation:.3f}")
                self.assertGreater(extreme, 3.0, f"{name}: 最前/最后权重比仅 {extreme:.2f}")

    def test_artifact_order_matches_model(self) -> None:
        """混音顺序必须与模型里的 ARTIFACT_NAMES 一致，否则 A 的列会错位。"""
        from unicore_eeg.spatial_mixer import ARTIFACT_MIXING_KINDS

        self.assertEqual(ARTIFACT_MIXING_KINDS, ARTIFACT_NAMES)

    def test_matrix_shape_and_column_semantics(self) -> None:
        matrix = self.mixer.build_matrix(torch.Generator().manual_seed(1))
        self.assertEqual(tuple(matrix.shape), (len(self.LAYOUT), len(ARTIFACT_NAMES)))
        self.assertFalse(torch.isnan(matrix).any())
        # 每列都归一化到最大绝对值 1，避免不同伪迹之间幅度尺度不可比
        for column in range(matrix.size(-1)):
            self.assertAlmostEqual(float(matrix[:, column].abs().max()), 1.0, places=5)

    def test_harmonic_prior_is_sparse(self) -> None:
        """harmonic：2–4 个通道非零（手册 §6.5）。"""
        from unicore_eeg.spatial_mixer import HARMONIC_ACTIVE_RANGE

        for seed in range(20):
            weights = self.mixer.build_mixing_matrix("harmonic", torch.Generator().manual_seed(seed))
            active = int((weights != 0).sum())
            self.assertGreaterEqual(active, HARMONIC_ACTIVE_RANGE[0])
            self.assertLessEqual(active, HARMONIC_ACTIVE_RANGE[1])

    def test_ocular_prior_favours_frontal_channels(self) -> None:
        """ocular：前额（Fp1/Fp2/F7/F8）权重显著高于枕区（O1/O2）。"""
        weights = self._weights("ocular_drift")
        frontal = max(weights[self.LAYOUT.index(name)] for name in ("Fp1", "Fp2", "F7", "F8"))
        occipital = max(weights[self.LAYOUT.index(name)] for name in ("O1", "O2"))
        self.assertGreater(frontal, 5.0 * occipital, f"前额 {frontal:.3f} vs 枕区 {occipital:.3f}")

    def test_myogenic_prior_is_local(self) -> None:
        """myogenic：局域——最大值通道离其他通道有明显衰减。"""
        weights = torch.tensor(self._weights("myogenic"))
        peak = int(weights.argmax())
        distance = (self.coords - self.coords[peak][None, :]).norm(dim=-1)
        farthest = int(distance.argmax())
        self.assertGreater(float(weights[peak]), 4.0 * float(weights[farthest]))

    def test_cardiac_prior_is_diffuse_and_coherent(self) -> None:
        """cardiac：全通道同号（低秩），合成出来的各通道时序高度相关。"""
        weights = torch.tensor(self._weights("cardiac"))
        self.assertTrue(bool((weights > 0).all()), "心电伪迹各通道应同号")
        self.assertLess(float(weights.max() - weights.min()), 0.5, "心电幅度应弥散而非局域")

        sources = torch.zeros(len(ARTIFACT_NAMES), 2000)
        time = torch.arange(2000).float() / 500.0
        sources[ARTIFACT_NAMES.index("cardiac")] = torch.sin(2 * torch.pi * time / 0.9)
        mixed, _ = self.mixer.mix_sources(sources, torch.Generator().manual_seed(11))
        correlation = torch.corrcoef(mixed)
        off_diagonal = correlation[~torch.eye(len(self.LAYOUT), dtype=torch.bool)]
        self.assertGreater(float(off_diagonal.mean()), 0.8, f"通道间相关性 {off_diagonal.mean():.3f}")

    def test_motion_prior_covers_all_channels(self) -> None:
        weights = torch.tensor(self._weights("motion_transient"))
        self.assertTrue(bool((weights > 0).all()))

    def test_unknown_prior_is_sparse(self) -> None:
        for seed in range(10):
            weights = self.mixer.build_mixing_matrix("unknown", torch.Generator().manual_seed(seed))
            active = int((weights != 0).sum())
            self.assertGreaterEqual(active, 3)
            self.assertLessEqual(active, 8)

    def test_mixed_artifact_channels_differ(self) -> None:
        """同一个伪迹在不同通道上必须不同（旧实现只是缩放+时移复制）。

        注意不能拿"任意两个通道"作判据：稀疏先验（harmonic / unknown）与局域先验
        （myogenic）下，未参与的通道权重正好是 0，两个零通道当然相等——那是先验的
        正确表现，不是复制。所以判据取**权重最大的两个通道**。
        """
        sources = torch.randn(len(ARTIFACT_NAMES), 1000)
        mixed, matrix = self.mixer.mix_sources(sources, self.generator)
        self.assertEqual(tuple(mixed.shape), (len(self.LAYOUT), 1000))
        self.assertEqual(tuple(matrix.shape), (len(self.LAYOUT), len(ARTIFACT_NAMES)))

        for index, kind in enumerate(ARTIFACT_NAMES):
            with self.subTest(kind=kind):
                single = torch.zeros_like(sources)
                single[index] = sources[index]
                components, column = self.mixer.mix_sources(single, torch.Generator().manual_seed(5))
                top = torch.topk(column[:, index].abs(), 2).indices
                first, second = int(top[0]), int(top[1])
                self.assertGreater(float(column[first, index].abs()), 0.0)
                self.assertFalse(
                    torch.allclose(components[first], components[second]),
                    f"{kind} 在权重最大的两个通道上完全相同",
                )

    def test_ocular_row_directly(self) -> None:
        """手册 T1.4 的原文验收：C=8 时 ``artifacts[:, 1, 0] != artifacts[:, 1, 1]``。"""
        sources = torch.randn(len(ARTIFACT_NAMES), 1000)
        single = torch.zeros_like(sources)
        index = ARTIFACT_NAMES.index("ocular_drift")
        single[index] = sources[index]
        components, _ = self.mixer.mix_sources(single, torch.Generator().manual_seed(5))
        self.assertEqual(tuple(components.shape), (8, 1000))
        self.assertFalse(torch.allclose(components[:, 0], components[:, 1]))

    def test_clean_mix_is_not_a_copy(self) -> None:
        """清洁源的空间混合同样要逐通道不同，并且枕区占优。"""
        source = torch.randn(1000)
        mixed = self.mixer.mix_clean(source, torch.Generator().manual_seed(3))
        self.assertEqual(tuple(mixed.shape), (len(self.LAYOUT), 1000))
        self.assertFalse(torch.allclose(mixed[0], mixed[1]))
        gains = mixed.abs().mean(dim=-1)
        occipital = max(gains[self.LAYOUT.index(name)] for name in ("O1", "O2"))
        frontal = min(gains[self.LAYOUT.index(name)] for name in ("Fp1", "Fp2"))
        self.assertGreater(occipital, frontal, "alpha 节律应枕区占优")

    def test_module_level_entry_point(self) -> None:
        """手册要求的签名：build_mixing_matrix(kind, C, coords, generator)。"""
        from unicore_eeg.spatial_mixer import build_mixing_matrix, canonical_head_layout

        weights = build_mixing_matrix(
            "ocular_drift", len(self.LAYOUT), self.coords, torch.Generator().manual_seed(2)
        )
        self.assertEqual(tuple(weights.shape), (len(self.LAYOUT),))
        # coords=None 时用确定性半球布局，同样能出结果
        fallback = build_mixing_matrix("cardiac", 5, None, torch.Generator().manual_seed(2))
        self.assertEqual(tuple(fallback.shape), (5,))
        layout = canonical_head_layout(16)
        self.assertEqual(tuple(layout.shape), (16, 3))
        self.assertTrue(bool((layout[:, 2] >= 0).all()), "半球布局应全部在上半球")
        self.assertTrue(torch.allclose(layout.norm(dim=-1), torch.ones(16), atol=1e-5))

    def test_ds004784_frame_axes(self) -> None:
        """体模的坐标系不是 RAS，必须靠 frame_axes 声明才知道"前"在哪。"""
        from unicore_eeg.spatial_mixer import SpatialMixer

        spec = self.montage.load_montage("ds004784")
        self.assertEqual(spec.frame_axes["anterior"], [1.0, 0.0, 0.0])

        mixer = SpatialMixer(spec.coords, sample_rate=512, mask=spec.mask, frame_axes=spec.frame_axes)
        frontal = spec.coords[mixer.frontal_index]
        posterior = spec.coords[mixer.posterior_index]
        self.assertGreater(float(frontal[0]), float(posterior[0]))
        # 错用默认 RAS 时"前"会落到 y 轴上，选出来的前极点必然不是同一个点
        naive = SpatialMixer(spec.coords, sample_rate=512, mask=spec.mask)
        self.assertNotEqual(naive.frontal_index, mixer.frontal_index)


class SpatialProjectionTests(unittest.TestCase):
    """T1.3 验收：空间投影模块。"""

    COUNTS = (1, 2, 3, 34, 64, 128)

    def test_module_parameter_count_is_independent_of_channels(self) -> None:
        """Gate 1 的判据：``SpatialProjectionHead`` 参数量与 C 无关。"""
        from unicore_eeg.model import SpatialProjectionHead

        counts = {
            channels: count_parameters(SpatialProjectionHead(UniCOREEGConfig(in_channels=channels)))
            for channels in self.COUNTS
        }
        self.assertEqual(len(set(counts.values())), 1, f"参数量随 C 变化：{counts}")

        # 更强的判据：模块里不能存在"输入维度恰好等于通道数"的线性层
        head = SpatialProjectionHead(UniCOREEGConfig(in_channels=64))
        channel_like = {1, 2, 3, 34, 64, 128}
        for module in head.modules():
            if isinstance(module, torch.nn.Linear):
                self.assertNotIn(
                    module.in_features,
                    channel_like,
                    f"发现输入维度 {module.in_features} 与通道数重合的 Linear，可能把 C 写死了",
                )

    def test_model_forward_and_backward_all_counts(self) -> None:
        """C ∈ {1,2,3,34,64,128} 前向 + 反向均成功且无 NaN。"""
        from unicore_eeg.model import UniCOREEG

        for channels in self.COUNTS:
            with self.subTest(channels=channels):
                model = UniCOREEG(UniCOREEGConfig(in_channels=channels, sample_rate=500))
                y = torch.randn(2, channels, 1000, requires_grad=True)
                outputs = model(y)
                self.assertEqual(tuple(outputs["clean"].shape), (2, channels, 1000))
                self.assertEqual(tuple(outputs["spatial_weights"].shape), (2, channels, 1))
                outputs["clean"].square().mean().backward()
                self.assertFalse(torch.isnan(outputs["clean"]).any())
                self.assertFalse(torch.isnan(y.grad).any())

    def test_alpha_zero_is_identity(self) -> None:
        """α 初值为 0：打开 use_spatial 的模型与关掉它的模型逐键输出一致。"""
        from unicore_eeg.model import UniCOREEG

        torch.manual_seed(0)
        disabled = UniCOREEG(UniCOREEGConfig(in_channels=4, use_spatial=False)).eval()
        enabled = UniCOREEG(UniCOREEGConfig(in_channels=4, use_spatial=True)).eval()
        enabled.load_state_dict(disabled.state_dict(), strict=False)
        y = torch.randn(2, 4, 1000)

        baseline, with_spatial = disabled(y), enabled(y)
        self.assertEqual(float(enabled.spatial.alpha_raw.detach()), 0.0)
        for key, value in baseline.items():
            self.assertTrue(torch.allclose(value, with_spatial[key]), f"{key} 不一致")
        # 关闭时不应多出空间相关的键
        self.assertNotIn("spatial_weights", baseline)
        self.assertIn("spatial_weights", with_spatial)

    def test_modulation_reaches_experts_for_the_right_observations(self) -> None:
        """机制级验证：只有 1/3 两路观测被调制，且调制量恰为 ``1+α·w_s``。

        不去比较最终输出——初始化时专家对观测项的敏感度很小，输出差异会小到
        落进 ``allclose`` 的容差里，那样的用例既不稳也说明不了问题。
        直接抓"每个专家实际收到的观测"，才是 §6.4 步骤 6 的契约。
        """
        from unicore_eeg import montage as M
        from unicore_eeg.model import SPATIAL_MODULATED_INDICES, UniCOREEG

        torch.manual_seed(5)
        spec = M.load_montage("physiomotion")
        model = UniCOREEG(
            UniCOREEGConfig(in_channels=spec.channel_count, sample_rate=1000, use_spatial=True)
        ).eval()
        with torch.no_grad():
            model.spatial.alpha_raw.fill_(2.0)  # tanh(2) ≈ 0.964

        y = torch.randn(2, spec.channel_count, 1000)
        raw: dict[int, torch.Tensor] = {}

        def hook(index):
            def _capture(module, args, kwargs):
                raw[index] = (args[2] if len(args) > 2 else kwargs["observation"]).detach().clone()

            return _capture

        handles = [
            expert.register_forward_pre_hook(hook(index), with_kwargs=True)
            for index, expert in enumerate(model.experts)
        ]
        try:
            with torch.no_grad():
                y_norm, _, _, _ = model.robust_normalize(y)
                observations = model.observations(y_norm)
                weights = model.spatial(y_norm, spec.coords, spec.mask)
                modulation = model.spatial.build_modulation(weights, observations[0])
                model(y, coords=spec.coords, coords_mask=spec.mask)
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(set(raw), set(range(len(ARTIFACT_NAMES))))
        self.assertFalse(torch.allclose(modulation, torch.ones_like(modulation)))
        for index in range(len(ARTIFACT_NAMES)):
            received = raw[index]
            self.assertEqual(tuple(received.shape), tuple(observations[index].shape))
            if index in SPATIAL_MODULATED_INDICES:
                # 直接比对乘法结果，不做除法：观测里存在接近零的样本，除法会把噪声放大
                expected = observations[index] * modulation
                self.assertTrue(
                    torch.allclose(received, expected, atol=1e-5),
                    f"第 {index} 路（{ARTIFACT_NAMES[index]}）的调制量与 1+α·w_s 不符"
                    f"（最大偏差 {float((received - expected).abs().max())}）",
                )
            else:
                self.assertTrue(
                    torch.allclose(received, observations[index], atol=1e-6),
                    f"第 {index} 路（{ARTIFACT_NAMES[index]}）本不应被空间权重改动",
                )

    def test_build_modulation_math(self) -> None:
        """``1 + α·w_s``：α 有界、初值为 0。"""
        from unicore_eeg.model import SpatialProjectionHead

        head = SpatialProjectionHead(UniCOREEGConfig(in_channels=8, spatial_modulation_alpha_scale=0.5))
        w_s = torch.full((2, 8, 1), 0.5)
        self.assertTrue(torch.allclose(head.build_modulation(w_s, w_s), torch.ones(2, 8, 1)))
        with torch.no_grad():
            head.alpha_raw.fill_(100.0)
        modulation = head.build_modulation(w_s, w_s)
        self.assertAlmostEqual(float(modulation.max().detach()), 1.25, places=5)  # 0.5*tanh(100)*0.5

    def test_orthonormalize_is_numerically_stable(self) -> None:
        """回归：正交化必须逐列归一化。

        旧实现把未归一化的列当参考向量，残差会被指数放大
        （实测 4 列就从 337 涨到 3.4e22，随后除以 inf 得到 NaN）。
        """
        from unicore_eeg.model import _orthonormalize

        torch.manual_seed(0)
        block = torch.randn(2, 1000, 4)
        result = _orthonormalize(block)
        self.assertTrue(bool(torch.isfinite(result).all()))
        norms = result.norm(dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-4))
        gram = torch.einsum("bti,btj->bij", result, result)
        identity = torch.eye(4).expand_as(gram)
        self.assertLess(float((gram - identity).abs().max()), 1e-3)

        # 大范数输入同样不能溢出
        large = _orthonormalize(block * 1e6)
        self.assertTrue(bool(torch.isfinite(large).all()))

    def test_power_iteration_matches_exact_svd(self) -> None:
        """幂迭代求出的子空间要与精确 SVD 的前 4 个右奇异向量一致（主角度）。"""
        from unicore_eeg.model import SpatialProjectionHead

        torch.manual_seed(1)
        channels, length, rank = 12, 1000, 4
        time = torch.arange(length).float() / 500.0
        sources = torch.stack(
            [torch.sin(2 * np.pi * (7.0 + 3.0 * index) * time) for index in range(rank)]
        )
        mixing = torch.randn(channels, rank)
        y = mixing @ sources + 0.01 * torch.randn(channels, length)
        y = y.unsqueeze(0)  # (1, C, T)

        common = dict(in_channels=channels, spatial_pca_rank=rank, spatial_power_iterations=4)
        power = SpatialProjectionHead(UniCOREEGConfig(spatial_pca_mode="power", **common))
        exact = SpatialProjectionHead(UniCOREEGConfig(spatial_pca_mode="svd", **common))

        with torch.no_grad():
            a = power.component_directions(y)[0]
            b = exact.component_directions(y)[0]
        cosines = torch.linalg.svdvals(a @ b.T)
        self.assertGreater(
            float(cosines.min()), 0.98, f"子空间主角度余弦 {cosines.tolist()}，幂迭代未收敛"
        )

        # 与真实源子空间的关系：幂迭代结果应张成同一个 4 维子空间
        source_basis = sources / sources.norm(dim=-1, keepdim=True)
        self.assertGreater(float(torch.linalg.svdvals(a @ source_basis.T).min()), 0.9)

    def test_coordinate_space_options(self) -> None:
        """两种坐标空间都能前向：raw 取前两分量，disk 走等距方位投影。"""
        from unicore_eeg import montage as M
        from unicore_eeg.model import SpatialProjectionHead

        spec = M.load_montage("physiomotion")
        outputs = {}
        for space in ("raw", "disk"):
            head = SpatialProjectionHead(
                UniCOREEGConfig(in_channels=spec.channel_count, spatial_coord_space=space)
            )
            projected = head.project_coordinates(spec.coords.unsqueeze(0), spec.mask)
            self.assertEqual(tuple(projected.shape), (1, spec.channel_count, 2))
            self.assertTrue(bool(torch.isfinite(projected).all()))
            with torch.no_grad():
                weights = head(torch.randn(2, spec.channel_count, 1000), spec.coords, spec.mask)
            self.assertTrue(bool(torch.isfinite(weights).all()))
            outputs[space] = projected
        # 两条路径必须给出不同的二维编码，否则说明选项没接上
        self.assertFalse(torch.allclose(outputs["raw"], outputs["disk"]))

        with self.assertRaises(ValueError):
            SpatialProjectionHead(
                UniCOREEGConfig(in_channels=4, spatial_coord_space="nonsense")
            ).project_coordinates(torch.randn(1, 4, 3))

    def test_whole_model_parameter_count_is_independent_of_channels(self) -> None:
        """同一 state_dict 可用于所有 C，整模型参数量不得随 C 变化。"""
        from unicore_eeg.model import UniCOREEG

        counts = [count_parameters(UniCOREEG(UniCOREEGConfig(in_channels=c))) for c in self.COUNTS]
        self.assertEqual(len(set(counts)), 1, f"模型参数量随 C 变化：{dict(zip(self.COUNTS, counts))}")

    def test_one_model_instance_runs_all_channel_counts(self) -> None:
        """同一实例、同一组权重依次处理多种 C，不重建模型。"""
        from unicore_eeg.model import UniCOREEG

        torch.manual_seed(11)
        model = UniCOREEG(UniCOREEGConfig(in_channels=1)).eval()
        reference_state = {name: value.clone() for name, value in model.state_dict().items()}
        for channels in self.COUNTS:
            with self.subTest(channels=channels), torch.no_grad():
                output = model(torch.randn(1, channels, 1000))
                self.assertEqual(tuple(output["clean"].shape), (1, channels, 1000))
                self.assertTrue(torch.isfinite(output["clean"]).all())
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, reference_state[name]), f"forward 修改了参数 {name}")
        restored = UniCOREEG(UniCOREEGConfig(in_channels=128)).eval()
        restored.load_state_dict(reference_state, strict=True)
        with torch.no_grad():
            output = restored(torch.randn(1, 128, 1000))
        self.assertEqual(tuple(output["clean"].shape), (1, 128, 1000))

    def test_channel_permutation_equivariance(self) -> None:
        """重排输入通道及坐标后，输出应以同一排列重排。"""
        from unicore_eeg.model import UniCOREEG

        torch.manual_seed(12)
        model = UniCOREEG().eval()
        signal = torch.randn(1, 8, 1000)
        coords = torch.randn(8, 3)
        permutation = torch.randperm(8)
        inverse = torch.argsort(permutation)
        with torch.no_grad():
            original = model(signal, coords=coords)
            reordered = model(signal[:, permutation], coords=coords[permutation])
        self.assertTrue(torch.allclose(original["clean"], reordered["clean"][:, inverse], atol=5e-4, rtol=1e-4))
        self.assertTrue(torch.allclose(
            original["artifact_components_norm"],
            reordered["artifact_components_norm"][:, :, inverse], atol=5e-4, rtol=1e-4
        ))

    def test_variable_channel_batch_and_masked_loss(self) -> None:
        """不同 C 可在同一 batch 补齐训练，padding 不贡献输出或损失。"""
        from unicore_eeg.batching import collate_variable_channels
        from unicore_eeg.losses import UniCORELoss
        from unicore_eeg.model import UniCOREEG

        samples = [
            SyntheticEEGDataset(samples=1, channels=3, length=1000, seed=19)[0],
            SyntheticEEGDataset(samples=1, channels=8, length=1000, seed=20)[0],
        ]
        batch = collate_variable_channels(samples)
        model = UniCOREEG().train()
        outputs = model(
            batch["noisy"], metadata=batch["metadata"], disabled_experts=batch["disabled_experts"],
            coords=batch["coords"], channel_mask=batch["channel_mask"],
        )
        loss, _ = UniCORELoss()(outputs, batch)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(tuple(batch["channel_mask"].shape), (2, 8))
        self.assertTrue(torch.equal(outputs["clean"][0, 3:], torch.zeros_like(outputs["clean"][0, 3:])))


class SyntheticMixingTests(unittest.TestCase):
    """T1.4 验收：合成器真混音。"""

    #: 一个前/后对照布局，前额四导 + 枕区两导 + 中央两导
    LAYOUT = ("Fp1", "Fp2", "F7", "F8", "O1", "O2", "C3", "C4")

    def setUp(self) -> None:
        from unicore_eeg import montage as M

        coords, mask = M.resolve_montage(self.LAYOUT)
        self.coords, self.mask = coords, mask

    def test_public_source_splits_are_disjoint(self) -> None:
        """公共源池的 train/val/test 原始索引必须严格不相交。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "eegdenoisenet"
            root.mkdir(parents=True)
            shape = (20, 32)
            for name in ("EEG_all_epochs_512hz.npy", "EOG_all_epochs.npy", "EMG_all_epochs_512hz.npy"):
                np.save(root / name, np.zeros(shape, dtype=np.float32))
            manifests = {split: PublicSignalPools(directory, split=split).split_manifest() for split in ("train", "val", "test")}
            for kind in manifests["train"]:
                sets = [set(manifests[split][kind]) for split in manifests]
                expected = len(set.union(*sets))
                self.assertEqual(expected, 20 if kind != "ecg" else 0)
                self.assertTrue(all(not (sets[i] & sets[j]) for i in range(3) for j in range(i + 1, 3)))

    def _dataset(self, channels: int = 8, use_coords: bool = True, samples: int = 4):
        dataset = SyntheticEEGDataset(
            samples=samples,
            channels=channels,
            length=1000,
            sample_rate=500,
            seed=11,
            split="train",
            montage_coords=self.coords if use_coords else None,
        )
        return dataset

    def test_sample_contract(self) -> None:
        """样本字典新增 mixing_matrix 与 coords，artifacts 形状保持 (6, C, T)。"""
        dataset = self._dataset()
        sample = dataset[0]
        for key in ("noisy", "clean", "artifacts", "mixing_matrix", "coords", "metadata"):
            self.assertIn(key, sample, f"样本缺少 {key}")
        self.assertEqual(tuple(sample["artifacts"].shape), (len(ARTIFACT_NAMES), 8, 1000))
        # 手册 T1.4 步骤 2 明确写 (6, C)：行是伪迹族，与 artifacts 的第一维对齐
        self.assertEqual(tuple(sample["mixing_matrix"].shape), (len(ARTIFACT_NAMES), 8))
        self.assertEqual(tuple(sample["coords"].shape), (8, 3))
        self.assertTrue(torch.allclose(sample["coords"], self.coords))
        self.assertEqual(tuple(sample["noisy"].shape), (8, 1000))
        self.assertFalse(torch.isnan(sample["noisy"]).any())

    def test_channel_count_one_still_works(self) -> None:
        """C=1 退化路径必须仍然可用（坐标缺省时用确定性半球布局）。"""
        for use_coords in (True, False):
            with self.subTest(use_coords=use_coords):
                dataset = SyntheticEEGDataset(
                    samples=4,
                    channels=1,
                    length=1000,
                    sample_rate=500,
                    seed=11,
                    split="train",
                    montage_coords=self.coords[:1] if use_coords else None,
                )
                sample = dataset[0]
                self.assertEqual(tuple(sample["artifacts"].shape), (len(ARTIFACT_NAMES), 1, 1000))
                self.assertEqual(tuple(sample["mixing_matrix"].shape), (len(ARTIFACT_NAMES), 1))
                self.assertEqual(tuple(sample["clean"].shape), (1, 1000))

    def test_coordinate_mismatch_is_rejected(self) -> None:
        """坐标通道数与 channels 不符时必须报错，而不是静默错位。"""
        with self.assertRaises(ValueError):
            SyntheticEEGDataset(channels=4, montage_coords=self.coords)

    def test_artifact_components_differ_across_channels(self) -> None:
        """同一伪迹在不同通道上必须不同——旧实现只是缩放 + 时移复制。

        判据取**混音权重最大的两个通道**：稀疏先验（harmonic / unknown）与局域先验
        （myogenic）下，未参与的通道权重接近 0，两个近零通道当然"相等"，
        那是先验的正确表现，不是复制。
        """
        dataset = self._dataset()
        checked = 0
        for index in range(20):
            sample = dataset[index]
            for kind, name in enumerate(ARTIFACT_NAMES):
                component = sample["artifacts"][kind]
                if float(component.abs().max()) == 0.0:
                    continue  # 该样本没有这一类伪迹
                row = sample["mixing_matrix"][kind]
                first, second = (int(item) for item in torch.topk(row.abs(), 2).indices)
                if min(float(row[first].abs()), float(row[second].abs())) < 0.1:
                    continue  # 两个通道权重都太小，比较没有意义
                self.assertFalse(
                    torch.allclose(component[first], component[second]),
                    f"样本 {index} 的 {name} 在权重最大的两个通道（{first}/{second}）上完全相同",
                )
                checked += 1
        self.assertGreater(checked, 10, "样本里几乎没有伪迹，测试没测到东西")

    def test_mixing_matrix_records_what_was_applied(self) -> None:
        """混音矩阵必须与"实际施加"一致：某下标有能量 ⟺ 该行非零。

        这条钉住一个真实缺陷：held-out 族的能量会被重定向到下标 5，若照抄本次生成的
        A 的第 5 列，记下的会是 unknown 源那一列，与实际施加的权重完全不符。
        """
        dataset = self._dataset(samples=40)
        checked = 0
        for index in range(40):
            sample = dataset[index]
            for kind in range(len(ARTIFACT_NAMES)):
                active = float(sample["labels"][kind]) == 1.0
                row_nonzero = float(sample["mixing_matrix"][kind].abs().max()) > 0.0
                self.assertEqual(
                    active,
                    row_nonzero,
                    f"样本 {index} 的第 {kind} 路（{ARTIFACT_NAMES[kind]}）："
                    f"labels={active} 但混音行非零={row_nonzero}",
                )
                checked += 1
        self.assertGreater(checked, 100)

    def test_held_out_family_prior_is_carried_into_the_unknown_row(self) -> None:
        """held-out 族的空间先验必须体现在下标 5 的混音行上。

        这是上面那个缺陷的正面判据：`first_experiment` 模式下 condition 7–11 会把
        第 0–4 族分别留出（重定向到下标 5）。因此
        condition 8 → 留出 ocular，下标 5 的行必须仍是"前额优势"；
        condition 10 → 留出 cardiac，下标 5 的行必须仍是"全通道同号"。
        若照抄了 unknown 源那一列，这两条都会不成立。
        """
        dataset = SyntheticEEGDataset(
            samples=12,
            channels=8,
            length=1000,
            sample_rate=500,
            seed=11,
            mode="first_experiment",
            split="train",
            montage_coords=self.coords,
        )
        unknown = ARTIFACT_NAMES.index("unknown")

        ocular_episode = dataset[8]  # condition 8 → family 1（ocular）被留出
        self.assertTrue(bool(ocular_episode["disabled_experts"][1]))
        self.assertEqual(float(ocular_episode["labels"][unknown]), 1.0)
        row = ocular_episode["mixing_matrix"][unknown]
        frontal = max(float(row[self.LAYOUT.index(name)]) for name in ("Fp1", "Fp2", "F7", "F8"))
        occipital = max(float(row[self.LAYOUT.index(name)]) for name in ("O1", "O2"))
        self.assertGreater(frontal, 5.0 * occipital, "下标 5 的混音行没有带上 ocular 的前额先验")

        cardiac_episode = dataset[10]  # condition 10 → family 3（cardiac）被留出
        self.assertTrue(bool(cardiac_episode["disabled_experts"][3]))
        row = cardiac_episode["mixing_matrix"][unknown]
        self.assertTrue(bool((row > 0).all()) or bool((row < 0).all()), "下标 5 的混音行没有带上 cardiac 的弥散同号先验")

    def test_ocular_mixing_favours_frontal(self) -> None:
        """混音矩阵的 ocular 行：前额显著占优，但远端非零且保留空间梯度。

        远端非零是必须的：纯高斯 σ=0.35 会把它压到 1e-10，等于该族在枕区不存在，
        那样"按空间分布去除眼动"在后部就退化成"什么都不用做"。同时也不能用一个大
        下界把远端压平——那会丢掉梯度。两条一起钉住，防止参数被随手调回任一端。
        """
        from unicore_eeg.spatial_mixer import OCULAR_FLOOR

        dataset = self._dataset(samples=60)
        ocular = ARTIFACT_NAMES.index("ocular_drift")
        checked = 0
        for index in range(60):
            sample = dataset[index]
            if float(sample["labels"][ocular]) != 1.0:
                continue
            row = sample["mixing_matrix"][ocular]
            frontal = max(float(row[self.LAYOUT.index(name)]) for name in ("Fp1", "Fp2", "F7", "F8"))
            occipital = max(float(row[self.LAYOUT.index(name)]) for name in ("O1", "O2"))
            self.assertGreater(frontal, 5.0 * occipital, "前额应显著占优")
            self.assertGreater(occipital, 0.5 * OCULAR_FLOOR, "枕区不应被衰减到零")
            # 三段均值应单调下降：前额 > 中央 > 枕区
            central = sum(float(row[self.LAYOUT.index(n)]) for n in ("C3", "C4")) / 2.0
            mean_frontal = sum(float(row[self.LAYOUT.index(n)]) for n in ("Fp1", "Fp2", "F8")) / 3.0
            self.assertGreater(mean_frontal, central)
            self.assertGreater(central, occipital)
            checked += 1
        self.assertGreater(checked, 5, "没有样本激活 ocular，测试没测到东西")

    def test_cardiac_component_is_coherent(self) -> None:
        """cardiac：混音行全通道同号；真正激活时产出的时序在各通道高度相关（近似低秩）。"""
        dataset = self._dataset(samples=60)
        cardiac = ARTIFACT_NAMES.index("cardiac")
        found = 0
        for index in range(60):
            sample = dataset[index]
            if float(sample["labels"][cardiac]) != 1.0:
                continue
            row = sample["mixing_matrix"][cardiac]
            self.assertTrue(bool((row > 0).all()) or bool((row < 0).all()), "心电权重应同号")
            component = sample["artifacts"][cardiac]
            self.assertGreater(float(component.abs().max()), 0.0)
            correlation = torch.corrcoef(component)
            off_diagonal = correlation[~torch.eye(correlation.size(0), dtype=torch.bool)]
            self.assertGreater(float(off_diagonal.mean()), 0.8)
            found += 1
        self.assertGreaterEqual(found, 5, "没有一个样本激活了 cardiac，测试没测到东西")

    def test_clean_is_spatially_mixed(self) -> None:
        """清洁源也要逐通道不同（不能是同一源复制）。"""
        dataset = self._dataset()
        differing = 0
        for index in range(10):
            clean = dataset[index]["clean"]
            if not torch.allclose(clean[0], clean[1]):
                differing += 1
        self.assertEqual(differing, 10)

    def test_severity_matches_measured_ratio(self) -> None:
        """`_scale_to_snr` 的 SNR 逻辑作用在混音后的多通道伪迹上。"""
        dataset = self._dataset()
        found = 0
        for index in range(20):
            sample = dataset[index]
            clean_rms = sample["clean"].square().mean().sqrt().clamp_min(1e-8)
            # 只核对下标 0–4：它们各自只可能收到一个族的缩放结果。
            # 下标 5 可能同时收下 unknown 源与 held-out 族，severity 取的是两者的最大值，
            # 与"求和后实测比值"本来就不该相等。
            for kind in range(len(ARTIFACT_NAMES) - 1):
                if float(sample["labels"][kind]) == 0.0:
                    continue
                measured = sample["artifacts"][kind].square().mean().sqrt() / clean_rms
                self.assertAlmostEqual(
                    float(sample["severity"][kind]),
                    float(measured),
                    delta=0.02,
                    msg=f"样本 {index} 的 {ARTIFACT_NAMES[kind]} severity 与实测不符",
                )
                found += 1
        self.assertGreater(found, 5)


if __name__ == "__main__":
    unittest.main()
