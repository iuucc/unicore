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
from unicore_eeg.model import ARTIFACT_NAMES, SparseRouter, UniCOREEGConfig
from unicore_eeg.physiomotion import map_annotation
from unicore_eeg.runtime import (
    TritonMissingError,
    check_ddp_allowed,
    configure_runtime,
    dataloader_kwargs,
)
from unicore_eeg.synthetic import SyntheticEEGDataset


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


if __name__ == "__main__":
    unittest.main()
