from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "tools" / "run_radeon_pod_notebooks.py"
SPEC = importlib.util.spec_from_file_location("run_radeon_pod_notebooks", MODULE_PATH)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def success_result(count: int = 1):
    return {
        "outputs": [
            {"output_type": "stream", "name": "stdout", "text": "ok\n"}
        ],
        "execution_count": count,
        "reply_status": "ok",
    }


def error_result(count: int = 1):
    return {
        "outputs": [
            {
                "output_type": "error",
                "ename": "RuntimeError",
                "evalue": "temporary failure",
                "traceback": [],
            }
        ],
        "execution_count": count,
        "reply_status": "error",
    }


class FakeSocket:
    def close(self):
        pass


class ProtocolSocket:
    def __init__(self, reply_status="ok"):
        self.reply_status = reply_status
        self.messages = []

    def send(self, raw):
        request = json.loads(raw)
        msg_id = request["header"]["msg_id"]

        def message(kind, content):
            return json.dumps(
                {
                    "header": {"msg_type": kind},
                    "parent_header": {"msg_id": msg_id},
                    "content": content,
                }
            )

        self.messages = [
            message("execute_input", {"execution_count": 7}),
            message("stream", {"name": "stdout", "text": "hello\n"}),
            message(
                "execute_reply",
                {
                    "status": self.reply_status,
                    "execution_count": 7,
                },
            ),
            message("status", {"execution_state": "idle"}),
        ]

    def settimeout(self, _timeout):
        pass

    def recv(self):
        return self.messages.pop(0)


class FakeJupyter:
    def __init__(self):
        self.sessions = 0
        self.session_paths = []
        self.deleted = []
        self.interrupted = []

    def create_session(self, path):
        self.sessions += 1
        self.session_paths.append(path)
        return f"session-{self.sessions}", f"kernel-{self.sessions}"

    def connect_channels(self, kernel_id, timeout):
        return FakeSocket()

    def delete_session(self, session_id):
        self.deleted.append(session_id)

    def interrupt_kernel(self, kernel_id):
        self.interrupted.append(kernel_id)


def args():
    return SimpleNamespace(
        cell_timeout=30,
        notebook_timeout=300,
        retry_delay_seconds=0,
        echo_output=False,
        echo_traceback=False,
    )


def notebook(*sources: str):
    return {
        "cells": [
            {
                "cell_type": "code",
                "metadata": {"ci_original_cell_index": index},
                "execution_count": None,
                "outputs": [],
                "source": source,
            }
            for index, source in enumerate(sources)
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


class RadeonGlobalPodAPITests(unittest.TestCase):
    def client(self):
        return RUNNER.RadeonPodClient(
            "https://example.invalid/api/huggingface/notebooks",
            "user@example.com",
            "radeon-secret-token",
            "registry/image:test",
            hf_token="hf-secret-token",
        )

    def test_create_uses_native_notebook_and_passes_runtime_environment(self):
        client = self.client()
        with (
            mock.patch.object(
                client,
                "current",
                return_value={"status": "not_found"},
            ),
            mock.patch.object(
                client,
                "_request",
                return_value={"status": "allocating"},
            ) as request,
        ):
            client.create("org/model")

        payload = request.call_args.kwargs["payload"]
        self.assertEqual(
            payload["notebook_path"],
            "https://huggingface.co/org/model.ipynb",
        )
        self.assertEqual(payload["image"], "registry/image:test")
        self.assertEqual(payload["env"]["HF_TOKEN"], "hf-secret-token")
        self.assertEqual(payload["env"]["HF_ENDPOINT"], RUNNER.DEFAULT_HF_ENDPOINT)

    def test_create_refuses_to_replace_an_existing_user_pod(self):
        client = self.client()
        with mock.patch.object(
            client,
            "current",
            return_value={"status": "ready", "instance_id": "existing"},
        ):
            with self.assertRaisesRegex(
                RUNNER.RadeonPodError,
                "refusing to replace",
            ):
                client.create("org/model")

    def test_cleanup_refuses_to_delete_a_different_instance(self):
        client = self.client()
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state.json"
            RUNNER.write_owned_state(
                state,
                user_name=client.user_name,
                model_id="org/model",
                instance_id="owned-instance",
            )
            with (
                mock.patch.object(
                    client,
                    "current",
                    return_value={
                        "status": "ready",
                        "instance_id": "different-instance",
                    },
                ),
                mock.patch.object(client, "delete_current") as delete,
            ):
                with self.assertRaisesRegex(
                    RUNNER.RadeonPodError,
                    "different Radeon Pod",
                ):
                    RUNNER.cleanup_owned_pod(client, state, 30, 0)

            delete.assert_not_called()
            self.assertTrue(state.exists())


class CellRetryTests(unittest.TestCase):
    def test_execute_request_waits_for_reply_and_idle_and_collects_output(self):
        socket = ProtocolSocket()

        result = RUNNER.execute_cell(
            socket,
            "print('hello')",
            "session",
            30,
            RUNNER.time.monotonic() + 30,
        )

        self.assertEqual(result["reply_status"], "ok")
        self.assertEqual(result["execution_count"], 7)
        self.assertEqual(result["outputs"][0]["text"], "hello\n")

    def test_non_ok_execute_reply_becomes_a_retryable_cell_error(self):
        socket = ProtocolSocket(reply_status="abort")

        result = RUNNER.execute_cell(
            socket,
            "work()",
            "session",
            30,
            RUNNER.time.monotonic() + 30,
        )

        self.assertEqual(result["reply_status"], "abort")
        self.assertIn(
            "KernelExecutionAborted",
            RUNNER.cell_error({"outputs": result["outputs"]}),
        )

    def test_failed_cell_is_retried_then_execution_continues_to_tail(self):
        jupyter = FakeJupyter()
        calls = []
        responses = [
            success_result(1),
            error_result(2),
            success_result(3),
            success_result(4),
        ]

        def execute(_socket, code, *_args):
            calls.append(code)
            return responses.pop(0)

        with (
            mock.patch.object(RUNNER, "execute_cell", side_effect=execute),
            mock.patch.object(RUNNER.time, "sleep"),
            mock.patch.object(
                RUNNER.time,
                "monotonic",
                side_effect=[100.0, 140.0],
            ),
        ):
            result = RUNNER.execute_remote_notebook(
                jupyter,
                "cloud/model.ipynb",
                notebook("first", "downloads-model", "tail"),
                args(),
                lambda *_: None,
            )

        self.assertEqual(
            calls,
            ["first", "downloads-model", "downloads-model", "tail"],
        )
        self.assertEqual(result.attempts_by_cell, {0: 1, 1: 2, 2: 1})
        self.assertEqual(result.elapsed_seconds, 40.0)
        self.assertIsNone(result.run_error)
        self.assertEqual(jupyter.sessions, 1)
        self.assertEqual(jupyter.session_paths, ["cloud/model.ipynb"])

    def test_three_failures_stop_before_remaining_cells(self):
        jupyter = FakeJupyter()
        calls = []

        def execute(_socket, code, *_args):
            calls.append(code)
            return error_result(len(calls))

        with (
            mock.patch.object(RUNNER, "execute_cell", side_effect=execute),
            mock.patch.object(RUNNER.time, "sleep"),
        ):
            result = RUNNER.execute_remote_notebook(
                jupyter,
                "cloud/model.ipynb",
                notebook("failing-download", "must-not-run"),
                args(),
                lambda *_: None,
            )

        self.assertEqual(calls, ["failing-download"] * 3)
        self.assertIn("failed after 3 attempts", result.run_error)
        self.assertEqual(result.attempts_by_cell, {0: 3, 1: 0})

    def test_kernel_loss_recreates_session_and_replays_from_first_cell(self):
        jupyter = FakeJupyter()
        calls = []
        responses = [
            success_result(1),
            RUNNER.JupyterAPIError("kernel disappeared"),
            success_result(1),
            success_result(2),
            success_result(3),
        ]

        def execute(_socket, code, *_args):
            calls.append(code)
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

        with (
            mock.patch.object(RUNNER, "execute_cell", side_effect=execute),
            mock.patch.object(RUNNER.time, "sleep"),
        ):
            result = RUNNER.execute_remote_notebook(
                jupyter,
                "cloud/model.ipynb",
                notebook("state-setup", "model-cell", "tail"),
                args(),
                lambda *_: None,
            )

        self.assertEqual(
            calls,
            ["state-setup", "model-cell", "state-setup", "model-cell", "tail"],
        )
        self.assertEqual(result.kernel_session_attempts, 2)
        self.assertIsNone(result.run_error)


class SourcePolicyTests(unittest.TestCase):
    def test_controller_leaves_notebook_provisioning_to_radeon_global(self):
        source = MODULE_PATH.read_text()

        self.assertNotIn('["hf", "download"', source)
        self.assertNotIn("ci_model_download", source)
        self.assertNotIn("common.load_notebook", source)
        self.assertNotIn("common.normalize_notebook", source)
        self.assertNotIn("common.validate_plan", source)
        self.assertNotIn("common.prune_original_notebook_snapshots", source)
        self.assertNotIn("common.sync_original_notebook_snapshots", source)
        self.assertNotIn("upload_notebook", source)
        self.assertIn("model_download=notebook-native", source)
        self.assertIn("sys.executable,", source)
        self.assertNotIn('["jupyter", "nbconvert"', source)

    def test_cloud_notebook_path_comes_from_jupyter_url(self):
        access_url = (
            "https://example.invalid/instances/pod/lab/tree/folder/"
            "My%20Notebook.ipynb?token=secret"
        )

        path = RUNNER.notebook_path_from_jupyter_url(access_url)

        self.assertEqual(path, "folder/My Notebook.ipynb")

    def test_cloud_notebook_is_read_through_jupyter_contents_api(self):
        jupyter = RUNNER.JupyterClient(
            "https://example.invalid/instances/pod/lab/tree/model.ipynb?token=secret"
        )
        cloud_notebook = notebook("print('cloud managed')")
        with mock.patch.object(
            jupyter,
            "_request",
            return_value={
                "type": "notebook",
                "format": "json",
                "content": cloud_notebook,
            },
        ) as request:
            result = jupyter.get_notebook("folder/model.ipynb")

        self.assertIs(result, cloud_notebook)
        request.assert_called_once_with("GET", "contents/folder/model.ipynb")

    def test_known_secrets_are_redacted_recursively_from_artifact_values(self):
        secret = "hf_example_secret_value"
        value = {
            "outputs": [
                {"text": f"token={secret}"},
                {"traceback": [f"Authorization: Bearer {secret}"]},
            ]
        }
        with mock.patch.object(RUNNER.common, "SECRET_VALUES", {secret}):
            redacted = RUNNER.redact_json_secrets(value)

        self.assertNotIn(secret, str(redacted))
        self.assertIn("******", str(redacted))


class PodSummaryTests(unittest.TestCase):
    def test_summary_marks_download_as_in_notebook_and_reports_cell_retries(self):
        target = RUNNER.common.Target("org/model", "org__model.ipynb")
        report = RUNNER.common.make_report(
            target,
            "radeon-pod__org__model.ipynb",
            42.0,
            None,
            [{"index": 1, "status": "PASSED", "error": None}],
            None,
        )
        report["cell_execution_retries"] = 2

        with tempfile.TemporaryDirectory() as directory:
            RUNNER.common.write_summary(Path(directory), [report], "pod-policy")
            summary = (Path(directory) / "summary.md").read_text()

        self.assertIn("| Model | Download | Model Download Tries | Cell Retries |", summary)
        self.assertIn("`org/model` | in notebook | \\ | 2 |", summary)
        self.assertIn("| 1/0/1 | 42s |", summary)


if __name__ == "__main__":
    unittest.main()
