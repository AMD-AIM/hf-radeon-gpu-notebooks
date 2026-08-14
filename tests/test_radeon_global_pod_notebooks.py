from __future__ import annotations

import copy
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
            "https://example.invalid/api/service/notebooks",
            "user@example.com",
            "radeon-secret-token",
            "registry/image:test",
            hf_token="hf-secret-token",
        )

    def test_create_uses_native_notebook_with_default_resource_payload(self):
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
        self.assertEqual(payload["pod_type"], "hf")
        self.assertEqual(payload["resource_template"], "resource-16c55g1u")
        self.assertEqual(payload["instance_type"], "jupyter")
        self.assertNotIn("gpu_count", payload)
        self.assertNotIn("env", payload)
        self.assertRegex(
            request.call_args.kwargs["idempotency_key"],
            r"^hf-oneclick-ci-[0-9a-f-]{36}$",
        )

    def test_production_default_uses_current_image(self):
        self.assertEqual(
            RUNNER.DEFAULT_RADEON_IMAGE,
            "10.5.10.12:1808/radeon-cloud-global/"
            "huaggingface_for_amd_radeon:20260812",
        )

    def test_create_uses_high_memory_resource_for_qwen3_omni(self):
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
            client.create("Qwen/Qwen3-Omni-30B-A3B-Instruct")

        payload = request.call_args.kwargs["payload"]
        self.assertEqual(payload["resource_template"], "resource-16c110g1u")

    def test_wait_ready_opens_service_notebook_for_jupyter_url(self):
        client = self.client()
        with mock.patch.object(
            client,
            "open_current",
            return_value={
                "url": "https://handoff.invalid/one-time",
                "direct_url": "https://jupyter.invalid/lab/tree/model.ipynb",
                "one_time": True,
            },
        ) as open_current:
            ready = client.wait_ready(
                {"status": "ready", "instance_id": "instance-1"},
                timeout=30,
                poll_seconds=0,
            )

        self.assertEqual(
            ready["jupyter_url"],
            "https://jupyter.invalid/lab/tree/model.ipynb",
        )
        self.assertEqual(
            ready["jupyter_handoff_url"],
            "https://handoff.invalid/one-time",
        )
        open_current.assert_called_once_with()

    def test_open_current_uses_service_open_endpoint(self):
        client = self.client()
        with mock.patch.object(client, "_request", return_value={}) as request:
            client.open_current()

        request.assert_called_once_with(
            "POST",
            "/current/open",
            params={"user_name": "user@example.com"},
        )

    def test_request_sends_idempotency_header(self):
        client = self.client()
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b"{}"
        with mock.patch.object(
            RUNNER.urllib.request,
            "urlopen",
            return_value=response,
        ) as urlopen:
            client._request(
                "POST",
                payload={"user_name": "user@example.com"},
                idempotency_key="test-idempotency-key",
            )

        sent_request = urlopen.call_args.args[0]
        self.assertEqual(
            sent_request.get_header("Idempotency-key"),
            "test-idempotency-key",
        )

    def test_jupyter_client_redeems_one_time_handoff(self):
        handoff_url = "https://handoff.invalid/one-time"
        direct_url = "https://jupyter.invalid/lab/tree/model.ipynb"
        opener = mock.MagicMock()
        opener.open.return_value.__enter__.return_value.geturl.return_value = (
            direct_url
        )
        with mock.patch.object(
            RUNNER.urllib.request,
            "build_opener",
            return_value=opener,
        ):
            jupyter = RUNNER.JupyterClient(
                handoff_url,
                bootstrap_url=handoff_url,
            )

        self.assertEqual(jupyter.access_url, direct_url)
        sent_request = opener.open.call_args.args[0]
        self.assertEqual(sent_request.full_url, handoff_url)

    def test_jupyter_client_posts_fragment_handoff_and_visits_landing_page(self):
        handoff_url = (
            "https://jupyter.invalid/_auth/start#request=one-time-assertion"
        )
        direct_url = "https://jupyter.invalid/lab/tree/model.ipynb"
        handoff_response = mock.MagicMock()
        handoff_response.__enter__.return_value.geturl.return_value = direct_url
        landing_response = mock.MagicMock()
        opener = mock.MagicMock()
        opener.open.side_effect = [handoff_response, landing_response]
        with mock.patch.object(
            RUNNER.urllib.request,
            "build_opener",
            return_value=opener,
        ):
            jupyter = RUNNER.JupyterClient(
                handoff_url,
                bootstrap_url=handoff_url,
            )

        self.assertEqual(jupyter.access_url, direct_url)
        self.assertEqual(opener.open.call_count, 2)
        handoff_request = opener.open.call_args_list[0].args[0]
        self.assertEqual(
            handoff_request.full_url,
            "https://jupyter.invalid/_auth/start",
        )
        self.assertEqual(handoff_request.method, "POST")
        self.assertEqual(handoff_request.data, b"request=one-time-assertion")
        landing_request = opener.open.call_args_list[1].args[0]
        self.assertEqual(landing_request.full_url, direct_url)
        self.assertEqual(landing_request.get_method(), "GET")

    def test_jupyter_mutation_requests_include_origin_and_referer(self):
        opener = mock.MagicMock()
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b"{}"
        opener.open.return_value = response
        with mock.patch.object(
            RUNNER.urllib.request,
            "build_opener",
            return_value=opener,
        ):
            jupyter = RUNNER.JupyterClient(
                "https://jupyter.invalid/lab/tree/model.ipynb"
            )

        jupyter._request("POST", "sessions", {})
        request = opener.open.call_args.args[0]
        self.assertEqual(request.get_header("Origin"), "https://jupyter.invalid")
        self.assertEqual(
            request.get_header("Referer"),
            "https://jupyter.invalid/lab/tree/model.ipynb",
        )

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

    def test_create_waits_for_a_prior_pod_that_is_still_deleting(self):
        client = self.client()
        with (
            mock.patch.object(
                client,
                "current",
                side_effect=[
                    {"status": "reconciling", "instance_id": "prior"},
                    {"status": "not_found"},
                ],
            ),
            mock.patch.object(client, "wait_deleted") as wait_deleted,
            mock.patch.object(
                client,
                "_request",
                return_value={"status": "allocating"},
            ) as request,
        ):
            client.create("org/model")

        wait_deleted.assert_called_once_with(600, 5)
        self.assertEqual(request.call_args.args, ("POST",))

    def test_wait_deleted_retries_a_transient_status_request_failure(self):
        client = self.client()
        with (
            mock.patch.object(
                client,
                "current",
                side_effect=[
                    RUNNER.RadeonPodError("temporary disconnect"),
                    {"status": "terminating"},
                    {"status": "not_found"},
                ],
            ) as current,
            mock.patch.object(RUNNER.time, "sleep") as sleep,
        ):
            client.wait_deleted(30, 0)

        self.assertEqual(current.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_wait_deleted_requires_stable_not_found_after_status_rebound(self):
        client = self.client()
        with (
            mock.patch.object(
                client,
                "current",
                side_effect=[
                    {"status": "not_found"},
                    {"status": "ready", "instance_id": "rebounded"},
                    {"status": "not_found"},
                    {"status": "not_found"},
                    {"status": "not_found"},
                ],
            ) as current,
            mock.patch.object(RUNNER.time, "sleep") as sleep,
            mock.patch.object(
                RUNNER.time,
                "monotonic",
                side_effect=[0.0, 1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0],
            ),
        ):
            client.wait_deleted(30, 0, stability_seconds=2)

        self.assertEqual(current.call_count, 5)
        self.assertEqual(sleep.call_count, 4)

    def test_cleanup_can_retain_owned_state_for_final_verification(self):
        client = self.client()
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state.json"
            RUNNER.write_owned_state(
                state,
                user_name=client.user_name,
                model_id="org/model",
                instance_id="owned-instance",
            )
            with mock.patch.object(
                RUNNER,
                "cleanup_matching_current_pod",
                return_value=4.0,
            ) as cleanup:
                elapsed = RUNNER.cleanup_owned_pod(
                    client,
                    state,
                    600,
                    5,
                    remove_state=False,
                    stability_seconds=90,
                )

            self.assertTrue(state.exists())

        self.assertEqual(elapsed, 4.0)
        cleanup.assert_called_once_with(
            client,
            "owned-instance",
            600,
            5,
            stability_seconds=90,
        )

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

    def test_startup_reset_deletes_one_preexisting_pod_and_waits(self):
        client = self.client()
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state.json"
            state.write_text("stale state")
            with (
                mock.patch.object(
                    client,
                    "current",
                    return_value={
                        "status": "ready",
                        "instance_id": "leftover-instance",
                    },
                ),
                mock.patch.object(client, "delete_current") as delete,
                mock.patch.object(client, "wait_deleted") as wait_deleted,
                mock.patch.object(
                    RUNNER.time,
                    "monotonic",
                    side_effect=[100.0, 104.0],
                ),
            ):
                elapsed = RUNNER.reset_current_pod_before_run(
                    client, state, 600, 5
                )

            self.assertFalse(state.exists())

        self.assertEqual(elapsed, 4.0)
        delete.assert_called_once_with()
        wait_deleted.assert_called_once_with(600, 5)

    def test_startup_reset_is_a_noop_when_no_pod_exists(self):
        client = self.client()
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state.json"
            state.write_text("stale state")
            with (
                mock.patch.object(
                    client,
                    "current",
                    return_value={"status": "not_found"},
                ),
                mock.patch.object(client, "delete_current") as delete,
                mock.patch.object(client, "wait_deleted") as wait_deleted,
                mock.patch.object(
                    RUNNER.time,
                    "monotonic",
                    side_effect=[100.0, 100.5],
                ),
            ):
                elapsed = RUNNER.reset_current_pod_before_run(
                    client, state, 600, 5
                )

            self.assertFalse(state.exists())

        self.assertEqual(elapsed, 0.5)
        delete.assert_not_called()
        wait_deleted.assert_not_called()


class CellRetryTests(unittest.TestCase):
    def test_session_cleanup_retries_transient_jupyter_route_failures(self):
        jupyter = FakeJupyter()
        jupyter.delete_session = mock.Mock(
            side_effect=[
                RUNNER.JupyterAPIError("route unavailable"),
                RUNNER.JupyterAPIError("route unavailable"),
                None,
            ]
        )
        with mock.patch.object(RUNNER.time, "sleep") as sleep:
            error = RUNNER.close_kernel_session(
                jupyter,
                FakeSocket(),
                "session-1",
                retry_delay_seconds=5,
            )

        self.assertIsNone(error)
        self.assertEqual(jupyter.delete_session.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

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
        self.assertEqual(result.attempts_by_cell, {0: 2, 1: 1, 2: 1})
        self.assertIsNone(result.run_error)
        self.assertEqual(result.errors_by_cell, {})

    def test_kernel_loss_stops_if_the_old_session_cannot_be_deleted(self):
        jupyter = FakeJupyter()
        jupyter.delete_session = mock.Mock(
            side_effect=RUNNER.JupyterAPIError("route unavailable")
        )
        calls = []

        def execute(_socket, code, *_args):
            calls.append(code)
            if code == "model-cell":
                raise RUNNER.JupyterAPIError("kernel channel closed")
            return success_result(1)

        with (
            mock.patch.object(RUNNER, "execute_cell", side_effect=execute),
            mock.patch.object(RUNNER, "SESSION_CLEANUP_ATTEMPTS", 2),
            mock.patch.object(RUNNER.time, "sleep"),
        ):
            result = RUNNER.execute_remote_notebook(
                jupyter,
                "cloud/model.ipynb",
                notebook("setup-cell", "model-cell", "must-not-run"),
                args(),
                lambda *_: None,
            )

        self.assertEqual(calls, ["setup-cell", "model-cell"])
        self.assertEqual(jupyter.sessions, 1)
        self.assertEqual(jupyter.delete_session.call_count, 2)
        self.assertIn("session cleanup failed after 2 attempts", result.run_error)

    def test_kernel_transport_failure_is_not_reported_as_a_passed_cell(self):
        jupyter = FakeJupyter()

        with (
            mock.patch.object(
                RUNNER,
                "execute_cell",
                side_effect=[
                    success_result(1),
                    RUNNER.JupyterAPIError(
                        "kernel channel closed before the cell became idle"
                    ),
                ],
            ),
            mock.patch.object(RUNNER, "KERNEL_SESSION_ATTEMPTS", 1),
        ):
            result = RUNNER.execute_remote_notebook(
                jupyter,
                "cloud/model.ipynb",
                notebook("setup-cell", "model-cell"),
                args(),
                lambda *_: None,
            )

        cells, passed, failed = RUNNER.collect_user_cell_results(
            result.notebook,
            result.attempts_by_cell,
            result.errors_by_cell,
        )

        self.assertEqual(passed, 1)
        self.assertEqual(failed, 1)
        self.assertEqual(
            [cell["status"] for cell in cells],
            ["PASSED", "FAILED"],
        )
        self.assertIn("kernel channel closed", cells[1]["error"])
        self.assertEqual(result.attempts_by_cell, {0: 1, 1: 1})
        self.assertIn("kernel channel closed", result.run_error)

        report = RUNNER.common.make_report(
            RUNNER.common.Target("org/model", "org__model.ipynb"),
            "radeon-pod__org__model.ipynb",
            result.elapsed_seconds,
            result.run_error,
            cells,
            result.timed_started_at,
        )
        report["cell_execution_retries"] = 0
        with tempfile.TemporaryDirectory() as directory:
            RUNNER.common.write_summary(Path(directory), [report], "pod-policy")
            summary = (Path(directory) / "summary.md").read_text()

        self.assertIn("| 1/1/2 |", summary)

    def test_session_handshake_failures_do_not_consume_cell_attempts(self):
        jupyter = FakeJupyter()
        create_session = mock.Mock(
            side_effect=[
                RUNNER.JupyterAPIError("still starting"),
                RUNNER.JupyterAPIError("still starting"),
                ("session-3", "kernel-3"),
            ]
        )
        jupyter.create_session = create_session

        with (
            mock.patch.object(RUNNER, "execute_cell", return_value=success_result()),
            mock.patch.object(RUNNER.time, "sleep"),
        ):
            result = RUNNER.execute_remote_notebook(
                jupyter,
                "cloud/model.ipynb",
                notebook("model-cell"),
                args(),
                lambda *_: None,
            )

        self.assertEqual(create_session.call_count, 3)
        self.assertEqual(result.kernel_session_attempts, 3)
        self.assertEqual(result.attempts_by_cell, {0: 1})
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

    def test_remote_inference_section_is_skipped_without_mutating_notebook(self):
        cloud_notebook = {
            "cells": [
                {"cell_type": "code", "source": "local_model()"},
                {
                    "cell_type": "markdown",
                    "source": "## Remote Inference via Inference Providers",
                },
                {
                    "cell_type": "code",
                    "source": "HF_TOKEN = 'YOUR_TOKEN_HERE'",
                },
                {"cell_type": "code", "source": "from openai import OpenAI"},
                {"cell_type": "markdown", "source": "## Another Local Section"},
                {"cell_type": "code", "source": "local_tail()"},
            ]
        }
        original = copy.deepcopy(cloud_notebook)

        selected, skipped = RUNNER.local_inference_code_cell_indexes(
            cloud_notebook
        )

        self.assertEqual(selected, [0, 5])
        self.assertEqual(skipped, [2, 3])
        self.assertEqual(cloud_notebook, original)

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

    def test_kernel_runtime_is_configured_without_mutating_notebook(self):
        cloud_notebook = notebook("print('user cell')")
        original = copy.deepcopy(cloud_notebook)
        calls = []

        def execute(_socket, code, *_args):
            calls.append(code)
            return success_result(len(calls))

        with mock.patch.object(RUNNER, "execute_cell", side_effect=execute):
            result = RUNNER.execute_remote_notebook(
                FakeJupyter(),
                "cloud/model.ipynb",
                cloud_notebook,
                args(),
                lambda *_: None,
                kernel_environment={
                    "HF_TOKEN": "hf-secret-token",
                },
            )

        self.assertEqual(len(calls), 2)
        self.assertIn("HF_TOKEN", calls[0])
        self.assertNotIn("HF_ENDPOINT", calls[0])
        self.assertEqual(calls[1], "print('user cell')")
        self.assertEqual(result.attempts_by_cell, {0: 1})
        self.assertEqual(cloud_notebook, original)

    def test_artifact_sanitization_drops_invalid_output_without_cloud_mutation(self):
        cloud_notebook = notebook("print('ok')")
        cloud_notebook["cells"][0]["output"] = {"nonstandard": True}
        cloud_notebook["cells"][0].pop("outputs")
        cloud_notebook["cells"][0].pop("execution_count")
        original = copy.deepcopy(cloud_notebook)

        artifact = RUNNER.sanitize_artifact_notebook(cloud_notebook)

        self.assertNotIn("output", artifact["cells"][0])
        self.assertEqual(artifact["cells"][0]["outputs"], [])
        self.assertIsNone(artifact["cells"][0]["execution_count"])
        self.assertEqual(cloud_notebook, original)


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
        report["resource_template"] = "resource-16c55g1u"
        omni_target = RUNNER.common.Target(
            "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            "Qwen__Qwen3-Omni-30B-A3B-Instruct.ipynb",
        )
        omni_report = RUNNER.common.make_report(
            omni_target,
            "radeon-pod__Qwen__Qwen3-Omni-30B-A3B-Instruct.ipynb",
            84.0,
            None,
            [{"index": 1, "status": "PASSED", "error": None}],
            None,
        )
        omni_report["cell_execution_retries"] = 0
        omni_report["resource_template"] = "resource-16c110g1u"

        with tempfile.TemporaryDirectory() as directory:
            RUNNER.common.write_summary(
                Path(directory), [report, omni_report], "pod-policy"
            )
            summary = (Path(directory) / "summary.md").read_text()

        self.assertIn(
            "| Model | Resource template | Download | Model Download Tries |",
            summary,
        )
        self.assertIn(
            "`org/model` | `resource-16c55g1u` | in notebook | \\ | 2 |",
            summary,
        )
        self.assertIn(
            "`Qwen/Qwen3-Omni-30B-A3B-Instruct` | "
            "`resource-16c110g1u` | in notebook | \\ | 0 |",
            summary,
        )
        self.assertIn("| 1/0/1 | 42s |", summary)

    def test_saved_report_records_selected_resource_template(self):
        target = RUNNER.common.Target(
            "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            "Qwen__Qwen3-Omni-30B-A3B-Instruct.ipynb",
        )
        with tempfile.TemporaryDirectory() as directory:
            report = RUNNER.save_report(
                target,
                Path(directory),
                "radeon-pod__Qwen__Qwen3-Omni-30B-A3B-Instruct.ipynb",
                None,
                None,
                None,
                None,
                0.0,
                0.0,
            )

        self.assertEqual(report["resource_template"], "resource-16c110g1u")


if __name__ == "__main__":
    unittest.main()
