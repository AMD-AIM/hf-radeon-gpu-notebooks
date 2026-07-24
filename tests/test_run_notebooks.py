from __future__ import annotations

import importlib.util
import os
import stat
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "tools" / "run_notebooks.py"
SPEC = importlib.util.spec_from_file_location("run_notebooks", MODULE_PATH)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


@contextmanager
def fake_notebook_modules(events: list[str]):
    nbformat = types.ModuleType("nbformat")
    nbformat.from_dict = lambda notebook: notebook
    nbformat.write = lambda notebook, path: Path(path).write_text("executed")

    class FakeNotebookClient:
        def __init__(self, notebook, **kwargs):
            self.notebook = notebook
            self.kwargs = kwargs
            self.km = None

        def execute(self):
            events.append("nbclient")
            return self.notebook

    nbclient = types.ModuleType("nbclient")
    nbclient.NotebookClient = FakeNotebookClient
    with mock.patch.dict(sys.modules, {"nbformat": nbformat, "nbclient": nbclient}):
        yield


class CacheConfigurationTests(unittest.TestCase):
    def test_shared_cache_accepts_one_hub_path(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            environment = {
                "HF_HOME": directory,
                "HF_HUB_CACHE": str(hub),
                "TRANSFORMERS_CACHE": str(hub),
            }
            with mock.patch.dict(os.environ, environment, clear=True):
                self.assertEqual(RUNNER.assert_shared_model_cache(), hub.resolve())

    def test_shared_cache_rejects_split_cli_and_transformers_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = {
                "HF_HOME": directory,
                "HF_HUB_CACHE": str(Path(directory) / "hub"),
                "TRANSFORMERS_CACHE": str(Path(directory) / "transformers"),
            }
            with mock.patch.dict(os.environ, environment, clear=True):
                with self.assertRaisesRegex(RuntimeError, "must share one cache"):
                    RUNNER.assert_shared_model_cache()


class ModelDownloadTests(unittest.TestCase):
    def test_plain_hf_download_retries_three_total_attempts(self):
        completed = [
            SimpleNamespace(returncode=1),
            SimpleNamespace(returncode=1),
            SimpleNamespace(returncode=0),
        ]
        with tempfile.TemporaryDirectory() as directory:
            environment = {
                "HF_HUB_CACHE": str(Path(directory) / "hub"),
                "TRANSFORMERS_CACHE": str(Path(directory) / "hub"),
            }
            with (
                mock.patch.dict(os.environ, environment, clear=True),
                mock.patch.object(RUNNER.subprocess, "run", side_effect=completed) as run,
                mock.patch.object(RUNNER.time, "sleep") as sleep,
                mock.patch.object(RUNNER.time, "monotonic", side_effect=[10.0, 22.0]),
            ):
                result = RUNNER.download_model("org/model")

        self.assertEqual(result["status"], "PASSED")
        self.assertEqual(result["attempts"], 3)
        self.assertEqual(result["elapsed_seconds"], 12.0)
        self.assertEqual(
            run.call_args_list,
            [mock.call(["hf", "download", "org/model"], check=False)] * 3,
        )
        self.assertEqual(sleep.call_count, 2)

    def test_real_subprocess_inherits_cache_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bin_dir = root / "bin"
            cache_dir = root / "cache" / "hub"
            bin_dir.mkdir()
            cache_dir.mkdir(parents=True)
            fake_hf = bin_dir / "hf"
            fake_hf.write_text(
                "#!/bin/sh\n"
                'printf "%s\\n" "$*" >> "$HF_HUB_CACHE/calls"\n'
                'printf "%s\\n" "$HF_HUB_CACHE" >> "$HF_HUB_CACHE/environments"\n'
                'count=$(wc -l < "$HF_HUB_CACHE/calls")\n'
                'if [ "$count" -lt 3 ]; then exit 1; fi\n'
            )
            fake_hf.chmod(fake_hf.stat().st_mode | stat.S_IXUSR)
            environment = {
                "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
                "HF_HUB_CACHE": str(cache_dir),
                "TRANSFORMERS_CACHE": str(cache_dir),
            }
            with (
                mock.patch.dict(os.environ, environment, clear=True),
                mock.patch.object(RUNNER.time, "sleep"),
            ):
                result = RUNNER.download_model("org/model")

            self.assertEqual(result["status"], "PASSED")
            self.assertEqual(result["attempts"], 3)
            self.assertEqual(
                (cache_dir / "calls").read_text().splitlines(),
                ["download org/model"] * 3,
            )
            self.assertEqual(
                (cache_dir / "environments").read_text().splitlines(),
                [str(cache_dir)] * 3,
            )


class SummaryTests(unittest.TestCase):
    def report(self, model_id: str, cache_mode: str, attempts: int, elapsed: float):
        with mock.patch.dict(os.environ, {"HF_CACHE_MODE": cache_mode}, clear=False):
            return RUNNER.make_report(
                RUNNER.Target(model_id=model_id, notebook=f"{model_id.split('/')[-1]}.ipynb"),
                "oneclick",
                f"oneclick__{model_id.split('/')[-1]}.ipynb",
                {"source": "snapshot", "fetched_url": None, "snapshot": "snapshot"},
                elapsed,
                {},
                None,
                [],
                download={
                    "status": "PASSED",
                    "attempts": attempts,
                    "elapsed_seconds": elapsed,
                    "cache_path": "/cache/hub",
                    "error": None,
                },
            )

    def test_summary_has_separate_tries_column_and_hides_runner_attempts(self):
        reports = [
            self.report("org/container-model", "container", 3, 12.0),
            self.report("org/runner-model", "runner", 1, 1.0),
        ]
        with tempfile.TemporaryDirectory() as directory:
            RUNNER.write_summary(Path(directory), reports, "test-policy")
            summary = (Path(directory) / "summary.md").read_text()

        self.assertIn("| Model | Download | Model Download Tries |", summary)
        self.assertIn("`org/container-model` | 12s | 3 |", summary)
        self.assertIn("`org/runner-model` | 1s | \\ |", summary)
        self.assertNotIn("(3x)", summary)


class ModelJobTests(unittest.TestCase):
    def args(self):
        return SimpleNamespace(
            cell_timeout=30,
            notebook_timeout=30,
            echo_output=False,
            echo_traceback=False,
        )

    def target(self):
        return RUNNER.Target(model_id="org/model", notebook="org__model.ipynb")

    def notebook(self):
        return {
            "cells": [
                {
                    "cell_type": "code",
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                    "source": "print('ok')",
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }

    def test_order_and_total_timer_cover_download_through_nbclient(self):
        events: list[str] = []
        download = {
            "status": "PASSED",
            "attempts": 2,
            "elapsed_seconds": 4.0,
            "cache_path": "/cache/hub",
            "error": None,
        }

        def fake_download(model_id, log=None):
            events.append("download")
            return download

        def fake_load(target):
            events.append("load-notebook")
            return self.notebook(), {
                "source": "snapshot",
                "fetched_url": None,
                "snapshot": "snapshot",
            }

        with tempfile.TemporaryDirectory() as directory, fake_notebook_modules(events):
            with (
                mock.patch.object(RUNNER, "download_model", side_effect=fake_download),
                mock.patch.object(RUNNER, "load_notebook", side_effect=fake_load),
                mock.patch.object(RUNNER, "poll_gpu"),
                mock.patch.object(
                    RUNNER.subprocess,
                    "run",
                    return_value=SimpleNamespace(returncode=0),
                ),
                mock.patch.object(
                    RUNNER.time,
                    "monotonic",
                    side_effect=[1.0, 6.0, 10.0, 15.0, 17.0, 18.0],
                ),
            ):
                report = RUNNER.run_one(self.target(), self.args(), Path(directory))

        self.assertEqual(events, ["load-notebook", "download", "nbclient"])
        self.assertEqual(report["overall_status"], "PASSED")
        self.assertEqual(report["download_attempts"], 2)
        self.assertEqual(report["download_elapsed_seconds"], 4.0)
        self.assertEqual(report["notebook_preparation_elapsed_seconds"], 5.0)
        self.assertEqual(report["notebook_elapsed_seconds"], 2.0)
        self.assertEqual(report["elapsed_seconds"], 8.0)

    def test_multiple_targets_run_as_download_notebook_pairs(self):
        events: list[str] = []

        def fake_download(model_id, log=None):
            events.append(f"download:{model_id}")
            return {
                "status": "PASSED",
                "attempts": 1,
                "elapsed_seconds": 1.0,
                "cache_path": "/cache/hub",
                "error": None,
            }

        def fake_load(target):
            events.append(f"load-notebook:{target.model_id}")
            return self.notebook(), {
                "source": "snapshot",
                "fetched_url": None,
                "snapshot": "snapshot",
            }

        targets = [
            RUNNER.Target(model_id="org/model-one", notebook="model-one.ipynb"),
            RUNNER.Target(model_id="org/model-two", notebook="model-two.ipynb"),
        ]
        with tempfile.TemporaryDirectory() as directory, fake_notebook_modules(events):
            with (
                mock.patch.object(RUNNER, "download_model", side_effect=fake_download),
                mock.patch.object(RUNNER, "load_notebook", side_effect=fake_load),
                mock.patch.object(RUNNER, "poll_gpu"),
                mock.patch.object(
                    RUNNER.subprocess,
                    "run",
                    return_value=SimpleNamespace(returncode=0),
                ),
            ):
                for target in targets:
                    RUNNER.run_one(target, self.args(), Path(directory))

        self.assertEqual(
            events,
            [
                "load-notebook:org/model-one",
                "download:org/model-one",
                "nbclient",
                "load-notebook:org/model-two",
                "download:org/model-two",
                "nbclient",
            ],
        )

    def test_exhausted_download_skips_notebook_execution_and_records_elapsed_time(self):
        events: list[str] = []
        download = {
            "status": "FAILED",
            "attempts": 3,
            "elapsed_seconds": 9.0,
            "cache_path": "/cache/hub",
            "error": "hf download exited with code 1",
        }

        def fake_download(model_id, log=None):
            events.append("download")
            return download

        def fake_load(target):
            events.append("load-notebook")
            return self.notebook(), {
                "source": "snapshot",
                "fetched_url": None,
                "snapshot": "snapshot",
            }

        with tempfile.TemporaryDirectory() as directory, fake_notebook_modules(events):
            with (
                mock.patch.object(RUNNER, "download_model", side_effect=fake_download),
                mock.patch.object(RUNNER, "load_notebook", side_effect=fake_load),
                mock.patch.object(
                    RUNNER.time,
                    "monotonic",
                    side_effect=[10.0, 12.0, 20.0, 29.0],
                ),
            ):
                report = RUNNER.run_one(self.target(), self.args(), Path(directory))

        self.assertEqual(events, ["load-notebook", "download"])
        self.assertEqual(report["overall_status"], "ERROR")
        self.assertEqual(report["download_attempts"], 3)
        self.assertEqual(report["notebook_preparation_elapsed_seconds"], 2.0)
        self.assertEqual(report["elapsed_seconds"], 9.0)
        self.assertIn("model download failed after 3 attempts", report["run_error"])

    def test_source_failure_is_excluded_and_prevents_model_download(self):
        events: list[str] = []

        def fake_load(target):
            events.append("load-notebook")
            raise RuntimeError("source unavailable")

        with tempfile.TemporaryDirectory() as directory, fake_notebook_modules(events):
            with (
                mock.patch.object(RUNNER, "load_notebook", side_effect=fake_load),
                mock.patch.object(RUNNER, "download_model") as download,
                mock.patch.object(
                    RUNNER.time,
                    "monotonic",
                    side_effect=[30.0, 35.0],
                ),
            ):
                report = RUNNER.run_one(self.target(), self.args(), Path(directory))

        self.assertEqual(events, ["load-notebook"])
        download.assert_not_called()
        self.assertEqual(report["overall_status"], "ERROR")
        self.assertEqual(report["notebook_preparation_elapsed_seconds"], 5.0)
        self.assertEqual(report["elapsed_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
