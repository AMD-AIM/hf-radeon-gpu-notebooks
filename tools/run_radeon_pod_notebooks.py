#!/usr/bin/env python3
"""Run Hugging Face One-Click notebooks in short-lived Radeon Cloud Pods."""

from __future__ import annotations

import argparse
import copy
import http.cookiejar
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    import radeon_global_ci_common as common
except ModuleNotFoundError:
    from tools import radeon_global_ci_common as common


DEFAULT_RADEON_API = "https://radeon-global.anruicloud.com/api/service/notebooks"
DEFAULT_RADEON_IMAGE = (
    "10.5.10.12:1808/radeon-cloud-global/"
    "huaggingface_for_amd_radeon:20260812"
)
DEFAULT_RESOURCE_TEMPLATE = "resource-16c55g1u"
HIGH_MEMORY_RESOURCE_TEMPLATE = "resource-16c110g1u"
HIGH_MEMORY_MODELS = frozenset({"Qwen/Qwen3-Omni-30B-A3B-Instruct"})
STATE_FILE = Path(".radeon-pod-ci-state.json")
CELL_EXECUTION_ATTEMPTS = 3
KERNEL_SESSION_ATTEMPTS = 5
REMOTE_INFERENCE_HEADING_RE = re.compile(
    r"(?im)^\s*##\s+Remote Inference via Inference Providers\b"
)
MARKDOWN_SECTION_HEADING_RE = re.compile(r"(?im)^\s*##\s+")


class RadeonPodError(RuntimeError):
    """A safe-to-log Radeon Pod lifecycle error."""


class JupyterAPIError(RuntimeError):
    """A safe-to-log Jupyter API or kernel transport error."""


@dataclass
class RemoteExecution:
    notebook: dict[str, Any]
    elapsed_seconds: float
    run_error: str | None
    attempts_by_cell: dict[int, int]
    errors_by_cell: dict[int, str]
    kernel_session_attempts: int
    timed_started_at: str | None
    timed_finished_at: str | None


def require_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RadeonPodError(f"{name} is required")
    if any(hint in name.upper() for hint in common.SECRET_HINTS):
        common.SECRET_VALUES.add(value)
    return value


def safe_response_text(raw: bytes, limit: int = 500) -> str:
    text = raw.decode("utf-8", errors="replace")
    return common.compact_error(common.redact_secrets(text), limit)


def find_jupyter_url(value: Any) -> str | None:
    if isinstance(value, dict):
        for key in (
            "jupyter_lab_url",
            "jupyter_url",
            "lab_url",
            "access_url",
            "url",
        ):
            candidate = value.get(key)
            if (
                isinstance(candidate, str)
                and candidate.startswith(("http://", "https://"))
                and "huggingface.co/" not in candidate
            ):
                return candidate
        for child in value.values():
            candidate = find_jupyter_url(child)
            if candidate:
                return candidate
    elif isinstance(value, list):
        for child in value:
            candidate = find_jupyter_url(child)
            if candidate:
                return candidate
    return None


def notebook_path_from_jupyter_url(access_url: str) -> str | None:
    parsed = urllib.parse.urlsplit(access_url)
    for marker in ("/lab/tree/", "/tree/", "/notebooks/"):
        if marker not in parsed.path:
            continue
        candidate = urllib.parse.unquote(parsed.path.split(marker, 1)[1]).lstrip("/")
        parts = candidate.split("/")
        if (
            candidate.lower().endswith(".ipynb")
            and all(part not in {"", ".", ".."} for part in parts)
        ):
            return candidate
    return None


def response_instance_id(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    # Do not treat a generic nested "id" as the Pod instance id: API payloads
    # may contain unrelated resource ids, which would make ownership cleanup
    # refuse the actual Pod.
    for key in ("instance_id", "pod_id"):
        candidate = value.get(key)
        if candidate not in (None, ""):
            return str(candidate)
    for child in value.values():
        candidate = response_instance_id(child)
        if candidate:
            return candidate
    return None


def response_status(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("status") or value.get("phase") or "").strip().lower()
    return ""


def resource_template_for_model(model_id: str) -> str:
    if model_id in HIGH_MEMORY_MODELS:
        return HIGH_MEMORY_RESOURCE_TEMPLATE
    return DEFAULT_RESOURCE_TEMPLATE


class RadeonPodClient:
    def __init__(
        self,
        api_url: str,
        user_name: str,
        api_token: str,
        image: str,
        hf_token: str = "",
        request_timeout: int = 60,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.user_name = user_name
        self.api_token = api_token
        self.image = image
        self.hf_token = hf_token
        self.request_timeout = request_timeout
        common.SECRET_VALUES.add(api_token)
        if hf_token:
            common.SECRET_VALUES.add(hf_token)

    def _request(
        self,
        method: str,
        path: str = "",
        *,
        params: dict[str, str] | None = None,
        payload: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        url = self.api_url + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "hf-oneclick-radeon-pod-ci",
        }
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        request = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = safe_response_text(exc.read()) or exc.reason
            raise RadeonPodError(
                f"Radeon API {method} {path or '/'} failed with HTTP "
                f"{exc.code}: {detail}"
            ) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RadeonPodError(
                f"Radeon API {method} {path or '/'} failed: "
                f"{type(exc).__name__}: {common.redact_secrets(str(exc))}"
            ) from None

        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            raise RadeonPodError(
                f"Radeon API {method} {path or '/'} returned non-JSON data: "
                f"{safe_response_text(raw)}"
            ) from None
        if not isinstance(data, dict):
            raise RadeonPodError(
                f"Radeon API {method} {path or '/'} returned an unexpected "
                f"{type(data).__name__}"
            )
        return data

    def current(self) -> dict[str, Any]:
        return self._request(
            "GET",
            "/current",
            params={"user_name": self.user_name},
        )

    def create(self, model_id: str) -> dict[str, Any]:
        current = self.current()
        status = response_status(current)
        if status != "not_found":
            instance = response_instance_id(current) or "unknown"
            raise RadeonPodError(
                "refusing to replace an existing Radeon Pod: "
                f"status={status or 'unknown'}, instance_id={instance}"
            )

        resource_template = resource_template_for_model(model_id)
        return self._request(
            "POST",
            payload={
                "user_name": self.user_name,
                "notebook_path": f"https://huggingface.co/{model_id}.ipynb",
                "pod_type": "hf",
                "resource_template": resource_template,
                "instance_type": "jupyter",
                "image": self.image,
                "unlimited_credits": True,
            },
            idempotency_key=f"hf-oneclick-ci-{uuid.uuid4()}",
        )

    def open_current(self) -> dict[str, Any]:
        return self._request(
            "POST",
            "/current/open",
            params={"user_name": self.user_name},
        )

    def delete_current(self) -> dict[str, Any]:
        return self._request(
            "DELETE",
            "/current",
            params={"user_name": self.user_name},
        )

    def wait_ready(
        self,
        initial: dict[str, Any],
        timeout: float,
        poll_seconds: float,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        current = initial
        access_url = find_jupyter_url(initial)
        handoff_url: str | None = None
        last_status = ""
        while True:
            status = response_status(current)
            access_url = find_jupyter_url(current) or access_url
            if status != last_status:
                print(
                    f"[POD] status={status or 'unknown'} "
                    f"instance_id={response_instance_id(current) or 'pending'}",
                    flush=True,
                )
                last_status = status
            if status == "ready" and not access_url:
                opened = self.open_current()
                handoff_url = find_jupyter_url(opened)
                direct_url = opened.get("direct_url")
                if not isinstance(direct_url, str) or not direct_url.startswith(
                    ("http://", "https://")
                ):
                    direct_url = None
                access_url = direct_url or handoff_url
                if not access_url:
                    raise RadeonPodError(
                        "ready Radeon Pod open response did not provide a Jupyter URL"
                    )
            if status == "ready" and access_url:
                result = dict(current)
                result["jupyter_url"] = access_url
                if handoff_url:
                    result["jupyter_handoff_url"] = handoff_url
                return result
            if status in {"failed", "error"}:
                message = common.compact_error(str(current.get("message") or ""), 300)
                raise RadeonPodError(
                    f"Radeon Pod entered {status}: {message or 'no detail'}"
                )
            if time.monotonic() >= deadline:
                raise RadeonPodError(
                    f"Radeon Pod did not become ready within {int(timeout)}s "
                    f"(last status={status or 'unknown'})"
                )
            time.sleep(poll_seconds)
            current = self.current()

    def wait_deleted(self, timeout: float, poll_seconds: float) -> None:
        deadline = time.monotonic() + timeout
        last_status = ""
        while True:
            current = self.current()
            status = response_status(current)
            if status != last_status:
                print(f"[POD] delete status={status or 'unknown'}", flush=True)
                last_status = status
            if status == "not_found":
                return
            if time.monotonic() >= deadline:
                raise RadeonPodError(
                    f"Radeon Pod was not fully deleted within {int(timeout)}s "
                    f"(last status={status or 'unknown'})"
                )
            time.sleep(poll_seconds)


def write_owned_state(
    path: Path,
    *,
    user_name: str,
    model_id: str,
    instance_id: str | None,
) -> None:
    state = {
        "owned_by": "hf-oneclick-radeon-pod-ci",
        "user_name": user_name,
        "model_id": model_id,
        "instance_id": instance_id,
        "created_at": common.utc_now(),
    }
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    path.chmod(0o600)


def read_owned_state(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        state = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RadeonPodError(
            f"could not read Pod ownership state {path}: {type(exc).__name__}: {exc}"
        ) from None
    if not isinstance(state, dict) or state.get("owned_by") != "hf-oneclick-radeon-pod-ci":
        raise RadeonPodError(f"refusing to use unrecognized Pod ownership state {path}")
    return state


def cleanup_owned_pod(
    client: RadeonPodClient,
    state_path: Path,
    delete_timeout: float,
    poll_seconds: float,
) -> float:
    state = read_owned_state(state_path)
    if state is None:
        return 0.0
    if state.get("user_name") != client.user_name:
        raise RadeonPodError(
            "refusing Pod cleanup because state user does not match RADEON_USER_NAME"
        )

    elapsed = cleanup_matching_current_pod(
        client,
        str(state.get("instance_id") or ""),
        delete_timeout,
        poll_seconds,
    )
    state_path.unlink(missing_ok=True)
    return elapsed


def cleanup_matching_current_pod(
    client: RadeonPodClient,
    expected_instance: str,
    delete_timeout: float,
    poll_seconds: float,
) -> float:
    started = time.monotonic()
    current = client.current()
    if response_status(current) == "not_found":
        return round(time.monotonic() - started, 3)

    current_instance = response_instance_id(current) or ""
    if expected_instance and current_instance and expected_instance != current_instance:
        raise RadeonPodError(
            "refusing to delete a different Radeon Pod: "
            f"expected instance_id={expected_instance}, current={current_instance}"
        )

    client.delete_current()
    client.wait_deleted(delete_timeout, poll_seconds)
    return round(time.monotonic() - started, 3)


def reset_current_pod_before_run(
    client: RadeonPodClient,
    state_path: Path,
    delete_timeout: float,
    poll_seconds: float,
) -> float:
    """Delete one pre-existing Pod before the serial model run starts."""
    started = time.monotonic()
    current = client.current()
    status = response_status(current)
    if status == "not_found":
        print("[POD] startup reset: no existing instance", flush=True)
        elapsed = round(time.monotonic() - started, 3)
        state_path.unlink(missing_ok=True)
        return elapsed
    if not status:
        raise RadeonPodError(
            "startup reset could not determine the current Radeon Pod status"
        )

    instance_id = response_instance_id(current) or "unknown"
    print(
        f"[POD] startup reset: deleting existing instance "
        f"status={status}, instance_id={instance_id}",
        flush=True,
    )
    client.delete_current()
    client.wait_deleted(delete_timeout, poll_seconds)
    elapsed = round(time.monotonic() - started, 3)
    state_path.unlink(missing_ok=True)
    return elapsed


class JupyterClient:
    def __init__(
        self,
        access_url: str,
        request_timeout: int = 60,
        bootstrap_url: str | None = None,
    ) -> None:
        self.request_timeout = request_timeout
        self.cookie_jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.cookie_jar)
        )
        handoff_posted = False

        if bootstrap_url:
            common.SECRET_VALUES.add(bootstrap_url)
            original_bootstrap_url = bootstrap_url
            bootstrap_parts = urllib.parse.urlsplit(bootstrap_url)
            handoff_fragment = urllib.parse.parse_qs(
                bootstrap_parts.fragment,
                keep_blank_values=True,
            )
            bootstrap_data = None
            bootstrap_headers = {
                "Accept": "text/html,application/xhtml+xml",
                "User-Agent": "hf-oneclick-radeon-pod-ci",
            }
            if handoff_fragment:
                if (
                    len(handoff_fragment) != 1
                    or next(iter(handoff_fragment)) not in {"request", "ticket"}
                ):
                    raise JupyterAPIError(
                        "Radeon Jupyter handoff URL had an invalid fragment"
                    )
                handoff_field, handoff_values = next(iter(handoff_fragment.items()))
                if len(handoff_values) != 1 or not handoff_values[0]:
                    raise JupyterAPIError(
                        "Radeon Jupyter handoff URL had an empty fragment"
                    )
                handoff_value = handoff_values[0]
                common.SECRET_VALUES.add(handoff_value)
                bootstrap_url = urllib.parse.urlunsplit(
                    (
                        bootstrap_parts.scheme,
                        bootstrap_parts.netloc,
                        bootstrap_parts.path,
                        bootstrap_parts.query,
                        "",
                    )
                )
                bootstrap_data = urllib.parse.urlencode(
                    {handoff_field: handoff_value}
                ).encode("utf-8")
                bootstrap_headers["Content-Type"] = (
                    "application/x-www-form-urlencoded"
                )
                handoff_posted = True
            bootstrap_request = urllib.request.Request(
                bootstrap_url,
                data=bootstrap_data,
                method="POST" if bootstrap_data is not None else "GET",
                headers=bootstrap_headers,
            )
            try:
                with self.opener.open(
                    bootstrap_request,
                    timeout=self.request_timeout,
                ) as response:
                    redirected_url = response.geturl()
            except urllib.error.HTTPError as exc:
                detail = safe_response_text(exc.read()) or exc.reason
                raise JupyterAPIError(
                    "Radeon Jupyter handoff failed with HTTP "
                    f"{exc.code}: {detail}"
                ) from None
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                raise JupyterAPIError(
                    "Radeon Jupyter handoff failed: "
                    f"{type(exc).__name__}: "
                    f"{common.redact_secrets(str(exc))}"
                ) from None
            if access_url == original_bootstrap_url:
                access_url = redirected_url

        parsed = urllib.parse.urlsplit(access_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise JupyterAPIError("Radeon API returned an invalid Jupyter URL")

        marker = re.search(r"/(?:lab|tree|notebooks)(?:/|$)", parsed.path)
        base_path = parsed.path[: marker.start()] if marker else parsed.path
        self.scheme = parsed.scheme
        self.netloc = parsed.netloc
        self.base_path = base_path.rstrip("/")
        self.query = parsed.query
        self.origin = f"{parsed.scheme}://{parsed.netloc}"
        self.access_url = access_url

        if handoff_posted:
            landing_request = urllib.request.Request(
                access_url,
                headers={
                    "Accept": "text/html,application/xhtml+xml",
                    "Referer": self.origin + "/_auth/start",
                    "User-Agent": "hf-oneclick-radeon-pod-ci",
                },
            )
            try:
                with self.opener.open(
                    landing_request,
                    timeout=self.request_timeout,
                ) as response:
                    response.read()
            except urllib.error.HTTPError as exc:
                detail = safe_response_text(exc.read()) or exc.reason
                raise JupyterAPIError(
                    "Radeon Jupyter landing page failed with HTTP "
                    f"{exc.code}: {detail}"
                ) from None
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                raise JupyterAPIError(
                    "Radeon Jupyter landing page failed: "
                    f"{type(exc).__name__}: "
                    f"{common.redact_secrets(str(exc))}"
                ) from None

        common.SECRET_VALUES.add(access_url)
        for _, value in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True):
            if value:
                common.SECRET_VALUES.add(value)

    def _url(self, api_path: str, *, websocket: bool = False) -> str:
        suffix = "/api" + ("/" + api_path.lstrip("/") if api_path else "")
        scheme = (
            ("wss" if self.scheme == "https" else "ws")
            if websocket
            else self.scheme
        )
        return urllib.parse.urlunsplit(
            (scheme, self.netloc, self.base_path + suffix, self.query, "")
        )

    def _request(
        self,
        method: str,
        api_path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "hf-oneclick-radeon-pod-ci",
        }
        for cookie in self.cookie_jar:
            if cookie.name == "_xsrf":
                headers["X-XSRFToken"] = cookie.value
                break
        if method not in {"GET", "HEAD"}:
            headers["Origin"] = self.origin
            headers["Referer"] = self.access_url
        request = urllib.request.Request(
            self._url(api_path),
            data=body,
            method=method,
            headers=headers,
        )
        try:
            with self.opener.open(request, timeout=self.request_timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = safe_response_text(exc.read()) or exc.reason
            raise JupyterAPIError(
                f"Jupyter API {method} /api/{api_path} failed with HTTP "
                f"{exc.code}: {detail}"
            ) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise JupyterAPIError(
                f"Jupyter API {method} /api/{api_path} failed: "
                f"{type(exc).__name__}: {common.redact_secrets(str(exc))}"
            ) from None

        if not raw:
            return {}
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            raise JupyterAPIError(
                f"Jupyter API {method} /api/{api_path} returned non-JSON data"
            ) from None
        if not isinstance(result, dict):
            raise JupyterAPIError(
                f"Jupyter API {method} /api/{api_path} returned an unexpected "
                f"{type(result).__name__}"
            )
        return result

    def wait_available(self, timeout: float, poll_seconds: float) -> None:
        deadline = time.monotonic() + timeout
        last_error = ""
        while True:
            try:
                self._request("GET", "")
                return
            except JupyterAPIError as exc:
                last_error = str(exc)
            if time.monotonic() >= deadline:
                raise JupyterAPIError(
                    f"Jupyter API was not available within {int(timeout)}s: "
                    f"{common.compact_error(last_error, 300)}"
                )
            time.sleep(poll_seconds)

    def get_notebook(self, path: str) -> dict[str, Any]:
        encoded_path = urllib.parse.quote(path, safe="/")
        result = self._request("GET", f"contents/{encoded_path}")
        content = result.get("content")
        if result.get("type") != "notebook" or not isinstance(content, dict):
            raise JupyterAPIError(
                f"cloud-managed path {path!r} is not a Jupyter notebook"
            )
        if not isinstance(content.get("cells"), list):
            raise JupyterAPIError(
                f"cloud-managed notebook {path!r} has no cell list"
            )
        return content

    def wait_notebook(
        self,
        path: str,
        timeout: float,
        poll_seconds: float,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        last_error = ""
        while True:
            try:
                return self.get_notebook(path)
            except JupyterAPIError as exc:
                last_error = str(exc)
            if time.monotonic() >= deadline:
                raise JupyterAPIError(
                    f"cloud-managed notebook {path!r} was not available within "
                    f"{int(timeout)}s: {common.compact_error(last_error, 300)}"
                )
            time.sleep(poll_seconds)

    def create_session(self, path: str) -> tuple[str, str]:
        result = self._request(
            "POST",
            "sessions",
            {
                "path": path,
                "name": Path(path).name,
                "type": "notebook",
                "kernel": {"name": "python3"},
            },
        )
        session_id = str(result.get("id") or "")
        kernel = result.get("kernel") if isinstance(result.get("kernel"), dict) else {}
        kernel_id = str(kernel.get("id") or "")
        if not session_id or not kernel_id:
            raise JupyterAPIError("Jupyter session response did not contain kernel ids")
        return session_id, kernel_id

    def delete_session(self, session_id: str) -> None:
        self._request("DELETE", f"sessions/{urllib.parse.quote(session_id, safe='')}")

    def interrupt_kernel(self, kernel_id: str) -> None:
        self._request(
            "POST",
            f"kernels/{urllib.parse.quote(kernel_id, safe='')}/interrupt",
            {},
        )

    def connect_channels(self, kernel_id: str, timeout: float) -> Any:
        try:
            import websocket
        except ModuleNotFoundError:
            raise JupyterAPIError(
                "the controller image must provide websocket-client"
            ) from None
        url = self._url(
            f"kernels/{urllib.parse.quote(kernel_id, safe='')}/channels",
            websocket=True,
        )
        try:
            cookie = "; ".join(
                f"{item.name}={item.value}" for item in self.cookie_jar
            )
            return websocket.create_connection(
                url,
                timeout=timeout,
                origin=self.origin,
                enable_multithread=True,
                **({"cookie": cookie} if cookie else {}),
            )
        except Exception as exc:
            raise JupyterAPIError(
                "could not connect to Jupyter kernel channels: "
                f"{type(exc).__name__}: {common.redact_secrets(str(exc))}"
            ) from None


def message_type(message: dict[str, Any]) -> str:
    header = message.get("header")
    if isinstance(header, dict) and header.get("msg_type"):
        return str(header["msg_type"])
    return str(message.get("msg_type") or "")


def append_stream_output(outputs: list[dict[str, Any]], name: str, text: Any) -> None:
    rendered = "".join(text) if isinstance(text, list) else str(text or "")
    if (
        outputs
        and outputs[-1].get("output_type") == "stream"
        and outputs[-1].get("name") == name
    ):
        outputs[-1]["text"] = str(outputs[-1].get("text") or "") + rendered
    else:
        outputs.append({"output_type": "stream", "name": name, "text": rendered})


def execute_cell(
    websocket: Any,
    code: str,
    session_id: str,
    cell_timeout: float,
    overall_deadline: float,
) -> dict[str, Any]:
    msg_id = uuid.uuid4().hex
    request = {
        "header": {
            "msg_id": msg_id,
            "username": "hf-oneclick-radeon-pod-ci",
            "session": session_id,
            "date": common.utc_now(),
            "msg_type": "execute_request",
            "version": "5.3",
        },
        "parent_header": {},
        "metadata": {},
        "content": {
            "code": code,
            "silent": False,
            "store_history": True,
            "user_expressions": {},
            "allow_stdin": False,
            "stop_on_error": True,
        },
        "channel": "shell",
    }
    try:
        websocket.send(json.dumps(request))
    except Exception as exc:
        raise JupyterAPIError(
            f"could not send execute_request: {type(exc).__name__}: "
            f"{common.redact_secrets(str(exc))}"
        ) from None

    cell_deadline = min(time.monotonic() + cell_timeout, overall_deadline)
    outputs: list[dict[str, Any]] = []
    execution_count: int | None = None
    execute_reply: dict[str, Any] | None = None
    idle = False
    clear_on_next_output = False

    while not (idle and execute_reply is not None):
        remaining = cell_deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"cell timeout > {int(cell_timeout)}s")
        try:
            websocket.settimeout(max(0.1, remaining))
            raw = websocket.recv()
        except (socket.timeout, TimeoutError) as exc:
            raise TimeoutError(f"cell timeout > {int(cell_timeout)}s") from exc
        except Exception as exc:
            if type(exc).__name__ == "WebSocketTimeoutException":
                raise TimeoutError(f"cell timeout > {int(cell_timeout)}s") from exc
            raise JupyterAPIError(
                f"kernel channel receive failed: {type(exc).__name__}: "
                f"{common.redact_secrets(str(exc))}"
            ) from None
        if raw in (None, ""):
            raise JupyterAPIError("kernel channel closed before the cell became idle")
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8")
            except UnicodeDecodeError:
                raise JupyterAPIError(
                    "kernel returned an unsupported binary WebSocket frame"
                ) from None
        try:
            message = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(message, dict):
            continue
        parent = message.get("parent_header")
        if not isinstance(parent, dict) or parent.get("msg_id") != msg_id:
            continue

        kind = message_type(message)
        content = message.get("content")
        if not isinstance(content, dict):
            content = {}

        if kind == "status" and content.get("execution_state") == "idle":
            idle = True
        elif kind == "execute_input":
            value = content.get("execution_count")
            execution_count = value if isinstance(value, int) else execution_count
        elif kind == "stream":
            if clear_on_next_output:
                outputs.clear()
                clear_on_next_output = False
            append_stream_output(
                outputs,
                str(content.get("name") or "stdout"),
                content.get("text"),
            )
        elif kind in {"execute_result", "display_data", "update_display_data"}:
            if clear_on_next_output:
                outputs.clear()
                clear_on_next_output = False
            output_type = "display_data" if kind == "update_display_data" else kind
            output: dict[str, Any] = {
                "output_type": output_type,
                "data": content.get("data") or {},
                "metadata": content.get("metadata") or {},
            }
            if output_type == "execute_result":
                value = content.get("execution_count")
                if isinstance(value, int):
                    execution_count = value
                output["execution_count"] = execution_count
            outputs.append(output)
        elif kind == "error":
            if clear_on_next_output:
                outputs.clear()
                clear_on_next_output = False
            outputs.append(
                {
                    "output_type": "error",
                    "ename": str(content.get("ename") or "Error"),
                    "evalue": str(content.get("evalue") or ""),
                    "traceback": list(content.get("traceback") or []),
                }
            )
        elif kind == "clear_output":
            if content.get("wait"):
                clear_on_next_output = True
            else:
                outputs.clear()
        elif kind == "execute_reply":
            execute_reply = content
            value = content.get("execution_count")
            if isinstance(value, int):
                execution_count = value

    reply_status = str((execute_reply or {}).get("status") or "unknown")
    if reply_status == "error":
        if not any(output.get("output_type") == "error" for output in outputs):
            outputs.append(
                {
                    "output_type": "error",
                    "ename": str(execute_reply.get("ename") or "Error"),
                    "evalue": str(execute_reply.get("evalue") or ""),
                    "traceback": list(execute_reply.get("traceback") or []),
                }
            )
    elif reply_status != "ok":
        outputs.append(
            {
                "output_type": "error",
                "ename": "KernelExecutionAborted",
                "evalue": f"execute_reply status={reply_status}",
                "traceback": [],
            }
        )
    return {
        "outputs": outputs,
        "execution_count": execution_count,
        "reply_status": reply_status,
    }


def cell_error(cell: dict[str, Any]) -> str | None:
    for output in cell.get("outputs", []):
        if output.get("output_type") == "error":
            return f"{output.get('ename')}: {output.get('evalue')}"
    return None


def reset_notebook_execution(notebook: dict[str, Any]) -> None:
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None


def log_cell_attempt(
    emit: Callable[[str, bool], None],
    cell: dict[str, Any],
    cell_index: int,
    attempt: int,
    echo_output: bool,
    echo_traceback: bool,
) -> None:
    helper = bool(cell.get("metadata", {}).get("ci_preamble"))
    label = "CI preamble" if helper else f"cell {cell_index}"
    emit(f"\n----- {label}, attempt {attempt}/{CELL_EXECUTION_ATTEMPTS} -----", False)
    if not helper:
        emit("".join(cell.get("source", [])).rstrip(), False)
        emit("----- output -----", False)
    for output in cell.get("outputs", []):
        kind = output.get("output_type")
        if kind == "stream":
            text = output.get("text")
            rendered = "".join(text) if isinstance(text, list) else str(text or "")
            emit(rendered.rstrip(), echo_output and not helper)
        elif kind in {"execute_result", "display_data"}:
            data = output.get("data")
            text = data.get("text/plain", "") if isinstance(data, dict) else ""
            rendered = "".join(text) if isinstance(text, list) else str(text or "")
            emit(rendered.rstrip(), echo_output and not helper)
        elif kind == "error":
            error = common.compact_error(
                f"{output.get('ename')}: {output.get('evalue')}",
                260,
            )
            emit(f"[ERROR] {label}: {error}", True)
            if echo_traceback:
                for line in output.get("traceback", []):
                    emit(common.strip_ansi(str(line)).rstrip(), False)


def close_kernel_session(
    jupyter: JupyterClient,
    websocket: Any,
    session_id: str | None,
) -> str | None:
    if websocket is not None:
        try:
            websocket.close()
        except Exception:
            pass
    if session_id:
        try:
            jupyter.delete_session(session_id)
        except JupyterAPIError as exc:
            return f"Jupyter session cleanup failed: {exc}"
    return None


def local_inference_code_cell_indexes(
    notebook: dict[str, Any],
) -> tuple[list[int], list[int]]:
    """Select code cells outside optional remote inference sections."""
    selected: list[int] = []
    skipped: list[int] = []
    in_remote_section = False
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") == "markdown":
            source = "".join(cell.get("source", []))
            if REMOTE_INFERENCE_HEADING_RE.search(source):
                in_remote_section = True
            elif in_remote_section and MARKDOWN_SECTION_HEADING_RE.search(source):
                in_remote_section = False
            continue
        if cell.get("cell_type") != "code":
            continue
        if in_remote_section:
            skipped.append(index)
        else:
            selected.append(index)
    return selected, skipped


def configure_kernel_runtime(
    websocket: Any,
    session_id: str,
    *,
    environment: dict[str, str],
    cell_timeout: float,
    overall_deadline: float,
) -> None:
    """Set controller-owned runtime values without changing the notebook."""
    if not environment:
        return

    source = (
        "import os as _ci_os\n"
        f"_ci_environment = {environment!r}\n"
        "for _ci_key, _ci_value in _ci_environment.items():\n"
        "    _ci_os.environ[_ci_key] = _ci_value\n"
        "del _ci_key, _ci_value, _ci_environment, _ci_os\n"
    )
    try:
        result = execute_cell(
            websocket,
            source,
            session_id,
            cell_timeout,
            overall_deadline,
        )
    except TimeoutError as exc:
        raise JupyterAPIError(f"kernel runtime configuration timed out: {exc}") from None
    error = cell_error({"outputs": result["outputs"]})
    if error:
        raise JupyterAPIError(
            "could not configure the kernel runtime: "
            f"{common.redact_secrets(error)}"
        )


def execute_remote_notebook(
    jupyter: JupyterClient,
    remote_path: str,
    cloud_notebook: dict[str, Any],
    args: argparse.Namespace,
    emit: Callable[[str, bool], None],
    kernel_environment: dict[str, str] | None = None,
) -> RemoteExecution:
    # Keep execution outputs in a controller-side copy. The cloud-managed
    # notebook is neither normalized nor overwritten through the Contents API.
    notebook = copy.deepcopy(cloud_notebook)

    code_cell_indexes, skipped_remote_indexes = local_inference_code_cell_indexes(
        notebook
    )
    if skipped_remote_indexes:
        emit(f"# skipped_remote_inference_code_cells={len(skipped_remote_indexes)}")
    attempts_by_cell = {index: 0 for index in code_cell_indexes}
    errors_by_cell: dict[int, str] = {}
    if not code_cell_indexes:
        return RemoteExecution(
            notebook=notebook,
            elapsed_seconds=0.0,
            run_error=None,
            attempts_by_cell=attempts_by_cell,
            errors_by_cell=errors_by_cell,
            kernel_session_attempts=0,
            timed_started_at=None,
            timed_finished_at=None,
        )

    session_id: str | None = None
    kernel_id: str | None = None
    websocket = None
    kernel_session_attempts = 0
    position = 0
    started: float | None = None
    timed_finished: float | None = None
    timed_started_at: str | None = None
    timed_finished_at: str | None = None
    run_error: str | None = None

    try:
        while position < len(code_cell_indexes):
            if websocket is None:
                if kernel_session_attempts >= KERNEL_SESSION_ATTEMPTS:
                    run_error = (
                        f"kernel/session recovery exhausted "
                        f"{KERNEL_SESSION_ATTEMPTS} attempts"
                    )
                    break
                kernel_session_attempts += 1
                emit(
                    f"[KERNEL] session attempt {kernel_session_attempts}/"
                    f"{KERNEL_SESSION_ATTEMPTS}",
                    False,
                )
                try:
                    session_id, kernel_id = jupyter.create_session(remote_path)
                    websocket = jupyter.connect_channels(kernel_id, args.cell_timeout)
                    if kernel_environment:
                        configure_kernel_runtime(
                            websocket,
                            session_id,
                            environment=kernel_environment,
                            cell_timeout=args.cell_timeout,
                            overall_deadline=time.monotonic() + args.notebook_timeout,
                        )
                except JupyterAPIError as exc:
                    run_error = str(exc)
                    cleanup_error = close_kernel_session(jupyter, websocket, session_id)
                    if cleanup_error:
                        run_error = f"{run_error}; {cleanup_error}"
                    websocket = None
                    session_id = None
                    kernel_id = None
                    if kernel_session_attempts >= KERNEL_SESSION_ATTEMPTS:
                        break
                    time.sleep(args.retry_delay_seconds)
                    continue
                if started is None:
                    started = time.monotonic()
                    timed_started_at = common.utc_now()
                    emit(
                        "# kernel_runtime="
                        f"HF_TOKEN={'set' if common.runtime_hf_token() else 'missing'} "
                        "(controller configured; excluded from timing)",
                        False,
                    )
                    emit(f"# timed_model_job_started {timed_started_at}", False)
                elif position:
                    emit(
                        "[KERNEL] prior state was lost; restarting execution "
                        "from the first cell",
                        True,
                    )
                    reset_notebook_execution(notebook)
                    position = 0

            cell_index = code_cell_indexes[position]
            cell = notebook["cells"][cell_index]
            attempts_by_cell[cell_index] += 1
            attempt = attempts_by_cell[cell_index]
            if attempt > CELL_EXECUTION_ATTEMPTS:
                run_error = (
                    f"cell {position + 1} exhausted "
                    f"{CELL_EXECUTION_ATTEMPTS} attempts"
                )
                errors_by_cell[cell_index] = run_error
                break

            try:
                result = execute_cell(
                    websocket,
                    "".join(cell.get("source", [])),
                    session_id or "",
                    args.cell_timeout,
                    (started or time.monotonic()) + args.notebook_timeout,
                )
                cell["outputs"] = result["outputs"]
                cell["execution_count"] = result["execution_count"]
            except TimeoutError as exc:
                errors_by_cell[cell_index] = (
                    f"cell {position + 1} timed out: {exc}"
                )
                emit(
                    f"[RETRY] cell {position + 1} attempt {attempt}/"
                    f"{CELL_EXECUTION_ATTEMPTS} timed out: {exc}",
                    True,
                )
                if kernel_id:
                    try:
                        jupyter.interrupt_kernel(kernel_id)
                    except JupyterAPIError as interrupt_error:
                        emit(f"[RETRY] kernel interrupt failed: {interrupt_error}", True)
                if attempt >= CELL_EXECUTION_ATTEMPTS:
                    run_error = (
                        f"cell {position + 1} failed after {attempt} attempts: {exc}"
                    )
                    break
                time.sleep(args.retry_delay_seconds)
                continue
            except JupyterAPIError as exc:
                errors_by_cell[cell_index] = (
                    f"cell {position + 1} execution was interrupted: {exc}"
                )
                emit(
                    f"[KERNEL] transport failure while executing cell "
                    f"{position + 1}: {exc}",
                    True,
                )
                cleanup_error = close_kernel_session(jupyter, websocket, session_id)
                if cleanup_error:
                    emit(f"[KERNEL] {cleanup_error}", True)
                websocket = None
                session_id = None
                kernel_id = None
                run_error = str(exc)
                if kernel_session_attempts >= KERNEL_SESSION_ATTEMPTS:
                    break
                time.sleep(args.retry_delay_seconds)
                continue

            log_cell_attempt(
                emit,
                cell,
                position + 1,
                attempt,
                args.echo_output,
                args.echo_traceback,
            )
            error = cell_error(cell)
            if error:
                errors_by_cell[cell_index] = error
                if attempt >= CELL_EXECUTION_ATTEMPTS:
                    run_error = (
                        f"cell {position + 1} failed after {attempt} attempts: {error}"
                    )
                    break
                emit(
                    f"[RETRY] cell {position + 1} failed; rerunning this cell, "
                    "then continuing through the remaining cells",
                    True,
                )
                time.sleep(args.retry_delay_seconds)
                continue

            errors_by_cell.pop(cell_index, None)
            run_error = None
            position += 1
            if position == len(code_cell_indexes):
                timed_finished = time.monotonic()
                timed_finished_at = common.utc_now()
    finally:
        if started is not None and timed_finished is None:
            timed_finished = time.monotonic()
            timed_finished_at = common.utc_now()
        cleanup_error = close_kernel_session(jupyter, websocket, session_id)
        if cleanup_error:
            run_error = f"{run_error}; {cleanup_error}" if run_error else cleanup_error

    elapsed = (
        round((timed_finished or started) - started, 1)
        if started is not None
        else 0.0
    )
    return RemoteExecution(
        notebook=notebook,
        elapsed_seconds=elapsed,
        run_error=run_error,
        attempts_by_cell=attempts_by_cell,
        errors_by_cell=errors_by_cell,
        kernel_session_attempts=kernel_session_attempts,
        timed_started_at=timed_started_at,
        timed_finished_at=timed_finished_at,
    )


def collect_user_cell_results(
    notebook: dict[str, Any],
    attempts_by_cell: dict[int, int],
    errors_by_cell: dict[int, str],
) -> tuple[list[dict[str, Any]], int, int]:
    cells: list[dict[str, Any]] = []
    passed = failed = user_index = 0
    for notebook_index, cell in enumerate(notebook.get("cells", [])):
        if (
            cell.get("cell_type") != "code"
            or cell.get("metadata", {}).get("ci_preamble")
        ):
            continue
        user_index += 1
        attempts = attempts_by_cell.get(notebook_index, 0)
        if not attempts:
            continue
        error = errors_by_cell.get(notebook_index) or cell_error(cell)
        if error:
            failed += 1
            cells.append(
                {
                    "index": user_index,
                    "status": "FAILED",
                    "error": common.redact_secrets(error),
                    "attempts": attempts,
                }
            )
        else:
            passed += 1
            cells.append(
                {
                    "index": user_index,
                    "status": "PASSED",
                    "error": None,
                    "attempts": attempts,
                }
            )
    return cells, passed, failed


def redact_json_secrets(value: Any) -> Any:
    if isinstance(value, str):
        return common.redact_secrets(value)
    if isinstance(value, list):
        return [redact_json_secrets(item) for item in value]
    if isinstance(value, dict):
        return {key: redact_json_secrets(item) for key, item in value.items()}
    return value


def sanitize_artifact_notebook(notebook: dict[str, Any]) -> dict[str, Any]:
    """Return a valid, redacted artifact copy without altering cloud content."""
    artifact = redact_json_secrets(copy.deepcopy(notebook))
    for cell in artifact.get("cells", []):
        if isinstance(cell, dict):
            cell.pop("output", None)
            if cell.get("cell_type") == "code":
                cell.setdefault("outputs", [])
                cell.setdefault("execution_count", None)
    return artifact


def save_report(
    target: common.Target,
    results_dir: Path,
    artifact_name: str,
    remote_notebook_path: str | None,
    execution: RemoteExecution | None,
    run_error: str | None,
    started_at: str | None,
    pod_setup_elapsed: float,
    pod_delete_elapsed: float,
) -> dict[str, Any]:
    executed_notebook = execution.notebook if execution else {"cells": []}
    attempts = execution.attempts_by_cell if execution else {}
    errors = execution.errors_by_cell if execution else {}
    cells, _, _ = collect_user_cell_results(executed_notebook, attempts, errors)
    elapsed = execution.elapsed_seconds if execution else 0.0
    report = common.make_report(
        target,
        artifact_name,
        elapsed,
        common.redact_secrets(run_error) if run_error else None,
        cells,
        execution.timed_started_at if execution else started_at,
    )
    user_attempts = [int(cell.get("attempts") or 0) for cell in cells]
    report.update(
        {
            "resource_template": resource_template_for_model(target.model_id),
            "pod_setup_elapsed_seconds": pod_setup_elapsed,
            "pod_delete_elapsed_seconds": pod_delete_elapsed,
            "timing_scope": "first-kernel-cell-start-to-last-kernel-cell-idle",
            "remote_notebook_path": remote_notebook_path,
            "cell_execution_attempts": sum(attempts.values()),
            "cell_execution_retries": sum(max(0, value - 1) for value in attempts.values()),
            "max_user_cell_attempts": max(user_attempts, default=0),
            "kernel_session_attempts": (
                execution.kernel_session_attempts if execution else 0
            ),
            "timed_finished_at": (
                execution.timed_finished_at if execution else None
            ),
        }
    )
    common.save_json(
        results_dir / artifact_name.replace(".ipynb", ".json"),
        report,
    )

    if execution is not None:
        output_notebook = results_dir / artifact_name
        safe_executed_notebook = sanitize_artifact_notebook(execution.notebook)
        output_notebook.write_text(
            json.dumps(safe_executed_notebook, indent=1, ensure_ascii=False) + "\n"
        )
        conversion = subprocess.run(
            [
                sys.executable,
                "-m",
                "nbconvert",
                "--to",
                "html",
                str(output_notebook),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if conversion.returncode:
            print(
                "[ARTIFACT-WARNING] nbconvert failed: "
                f"{common.compact_error(conversion.stderr, 300)}",
                flush=True,
            )
    return report


def append_error(current: str | None, additional: str) -> str:
    return f"{current}; {additional}" if current else additional


def run_one(
    target: common.Target,
    args: argparse.Namespace,
    results_dir: Path,
    client: RadeonPodClient,
) -> dict[str, Any]:
    artifact_name = f"radeon-pod__{target.notebook}"
    log_path = results_dir / artifact_name.replace(".ipynb", ".log")
    remote_notebook_path: str | None = None
    execution: RemoteExecution | None = None
    run_error: str | None = None
    started_at: str | None = None
    pod_setup_elapsed = 0.0
    pod_delete_elapsed = 0.0
    pod_created = False
    created_instance_id = ""
    state_path = Path(args.state_file)

    with log_path.open("w", buffering=1) as log:
        def emit(line: str, stdout: bool = False) -> None:
            safe = common.redact_secrets(line)
            log.write(safe + "\n")
            if stdout:
                print(safe, flush=True)

        emit(f"# {artifact_name} ({target.model_id}) mode=radeon-pod")

        pod_setup_started = time.monotonic()
        try:
            create_response = client.create(target.model_id)
            pod_created = True
            created_instance_id = response_instance_id(create_response) or ""
            write_owned_state(
                state_path,
                user_name=client.user_name,
                model_id=target.model_id,
                instance_id=created_instance_id or None,
            )
            ready = client.wait_ready(
                create_response,
                args.pod_ready_timeout,
                args.pod_poll_seconds,
            )
            instance_id = response_instance_id(ready)
            created_instance_id = instance_id or created_instance_id
            write_owned_state(
                state_path,
                user_name=client.user_name,
                model_id=target.model_id,
                instance_id=created_instance_id or None,
            )
            access_url = find_jupyter_url(ready)
            if not access_url:
                raise RadeonPodError("ready Radeon Pod did not provide a Jupyter URL")
            jupyter = JupyterClient(
                access_url,
                request_timeout=args.jupyter_request_timeout,
                bootstrap_url=ready.get("jupyter_handoff_url"),
            )
            jupyter.wait_available(
                args.jupyter_ready_timeout,
                args.pod_poll_seconds,
            )
            remote_path = notebook_path_from_jupyter_url(jupyter.access_url)
            if not remote_path:
                raise RadeonPodError(
                    "ready Radeon Pod URL did not identify a cloud-managed notebook"
                )
            cloud_notebook = jupyter.wait_notebook(
                remote_path,
                args.jupyter_ready_timeout,
                args.pod_poll_seconds,
            )
            remote_notebook_path = remote_path
            pod_setup_elapsed = round(time.monotonic() - pod_setup_started, 3)
            emit(
                f"# pod_setup={pod_setup_elapsed}s (excluded) "
                f"instance_id={created_instance_id or 'unknown'} "
                f"resource_template={resource_template_for_model(target.model_id)}"
            )
            emit(f"# cloud_notebook={remote_path} (managed by Radeon Global)")

            execution = execute_remote_notebook(
                jupyter,
                remote_path,
                cloud_notebook,
                args,
                emit,
                kernel_environment={
                    **(
                        {
                            "HF_TOKEN": client.hf_token,
                            "HUGGING_FACE_HUB_TOKEN": client.hf_token,
                            "HUGGINGFACEHUB_API_TOKEN": client.hf_token,
                        }
                        if client.hf_token
                        else {}
                    ),
                },
            )
            started_at = execution.timed_started_at
            run_error = execution.run_error
        except (RadeonPodError, JupyterAPIError, OSError, ValueError) as exc:
            if not pod_setup_elapsed:
                pod_setup_elapsed = round(time.monotonic() - pod_setup_started, 3)
            run_error = append_error(
                run_error,
                f"{type(exc).__name__}: {common.redact_secrets(str(exc))}",
            )
            emit(f"[INFRASTRUCTURE-ERROR] {run_error}", True)
        finally:
            if state_path.exists():
                try:
                    pod_delete_elapsed = cleanup_owned_pod(
                        client,
                        state_path,
                        args.pod_delete_timeout,
                        args.pod_poll_seconds,
                    )
                    emit(f"# pod_delete={pod_delete_elapsed}s (excluded)")
                except RadeonPodError as exc:
                    cleanup_error = f"Pod cleanup failed: {exc}"
                    run_error = append_error(run_error, cleanup_error)
                    emit(f"[CLEANUP-ERROR] {cleanup_error}", True)
            elif pod_created:
                try:
                    pod_delete_elapsed = cleanup_matching_current_pod(
                        client,
                        created_instance_id,
                        args.pod_delete_timeout,
                        args.pod_poll_seconds,
                    )
                    emit(f"# pod_delete={pod_delete_elapsed}s (excluded)")
                except RadeonPodError as exc:
                    cleanup_error = f"Pod cleanup failed without state file: {exc}"
                    run_error = append_error(run_error, cleanup_error)
                    emit(f"[CLEANUP-ERROR] {cleanup_error}", True)

        report = save_report(
            target,
            results_dir,
            artifact_name,
            remote_notebook_path,
            execution,
            run_error,
            started_at,
            pod_setup_elapsed,
            pod_delete_elapsed,
        )
        emit(
            f"# RESULT {report['overall_status']} "
            f"cells={report['cells_passed']}/{report['cells_total']} "
            f"cell_retries={report['cell_execution_retries']} "
            f"timed_total={report['elapsed_seconds']}s "
            f"pod_setup={pod_setup_elapsed}s pod_delete={pod_delete_elapsed}s"
        )

    print(
        f"[{report['overall_status']:6}] radeon-pod {target.notebook:45} "
        f"cells {report['cells_passed']}/{report['cells_total']} "
        f"retries {report['cell_execution_retries']} "
        f"timed {report['elapsed_seconds']}s",
        flush=True,
    )
    return report


def build_client(args: argparse.Namespace) -> RadeonPodClient:
    token = require_environment("RADEON_API_TOKEN")
    user_name = require_environment("RADEON_USER_NAME")
    hf_token = common.runtime_hf_token()
    return RadeonPodClient(
        api_url=args.radeon_api,
        user_name=user_name,
        api_token=token,
        image=args.radeon_image,
        hf_token=hf_token,
        request_timeout=args.radeon_request_timeout,
    )


def validate_cloud_plan(targets: list[common.Target]) -> list[str]:
    errors: list[str] = []
    print(f"Plan contains {len(targets)} Radeon Global notebook job(s).", flush=True)
    for target in targets:
        notebook_url = f"https://huggingface.co/{target.model_id}.ipynb"
        if not target.model_id.strip() or not target.notebook.lower().endswith(".ipynb"):
            error = f"invalid target mapping: {target.model_id!r}, {target.notebook!r}"
            errors.append(error)
            print(f"[PLAN ERR] {error}", flush=True)
            continue
        print(
            f"[PLAN OK] radeon-pod {target.model_id:45} "
            f"resource_template={resource_template_for_model(target.model_id)} "
            f"artifact={target.notebook:45} cloud_source={notebook_url}",
            flush=True,
        )
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--target-file", default=str(common.TARGET_CSV))
    parser.add_argument("--filter", default="")
    parser.add_argument("--fail-on", choices=["all", "none"], default="all")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--cleanup-state", action="store_true")
    parser.add_argument("--state-file", default=str(STATE_FILE))
    parser.add_argument("--cell-timeout", type=int, default=1800)
    parser.add_argument("--notebook-timeout", type=int, default=10800)
    parser.add_argument("--retry-delay-seconds", type=float, default=5.0)
    parser.add_argument("--pod-ready-timeout", type=int, default=1800)
    parser.add_argument("--pod-delete-timeout", type=int, default=600)
    parser.add_argument("--pod-poll-seconds", type=float, default=5.0)
    parser.add_argument("--jupyter-ready-timeout", type=int, default=300)
    parser.add_argument("--jupyter-request-timeout", type=int, default=60)
    parser.add_argument("--radeon-request-timeout", type=int, default=60)
    parser.add_argument(
        "--radeon-api",
        default=os.environ.get("RADEON_NOTEBOOK_API", DEFAULT_RADEON_API),
    )
    parser.add_argument(
        "--radeon-image",
        default=os.environ.get("RADEON_POD_IMAGE", DEFAULT_RADEON_IMAGE),
    )
    parser.add_argument("--echo-output", action="store_true")
    parser.add_argument("--echo-traceback", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    target_file = Path(args.target_file)
    targets = common.load_targets(target_file, args.filter)
    if not targets and not args.cleanup_state:
        raise SystemExit(f"no enabled targets matched filter {args.filter!r}")

    if args.plan_only:
        errors = validate_cloud_plan(targets)
        raise SystemExit(1 if errors else 0)

    client = build_client(args)
    state_path = Path(args.state_file)
    if args.cleanup_state:
        cleanup_owned_pod(
            client,
            state_path,
            args.pod_delete_timeout,
            args.pod_poll_seconds,
        )
        return

    reset_elapsed = reset_current_pod_before_run(
        client,
        state_path,
        args.pod_delete_timeout,
        args.pod_poll_seconds,
    )
    print(f"[POD] startup reset completed in {reset_elapsed}s", flush=True)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    policy = (
        f"source=radeon-global-managed; fail_on={args.fail_on}; "
        "backend=radeon-pod; "
        "startup_current_pod_reset=true; "
        "pod_per_model=true; resource_template=per-model; "
        "model_download=notebook-native; "
        f"cell_attempts={CELL_EXECUTION_ATTEMPTS}; "
        f"kernel_session_attempts={KERNEL_SESSION_ATTEMPTS}; "
        "timing=first-kernel-cell-start-to-last-kernel-cell-idle; "
        "cloud-notebook-setup/pod-create/pod-delete=excluded"
    )
    reports: list[dict[str, Any]] = []
    pending = [f"radeon-pod__{target.notebook}" for target in targets]
    common.write_progress(results_dir, reports, pending)

    print(f"Running Radeon Pod notebook CI: targets={len(targets)}", flush=True)
    for target in targets:
        run_name = f"radeon-pod__{target.notebook}"
        pending.remove(run_name)
        print(
            f"==> START {run_name} ({target.model_id}): "
            f"create {resource_template_for_model(target.model_id)} Pod and use "
            "its cloud-managed notebook "
            "(excluded) -> "
            "notebook cells with retry (timed) -> delete Pod (excluded)",
            flush=True,
        )
        common.write_progress(results_dir, reports, pending, running=run_name)
        report = run_one(target, args, results_dir, client)
        reports.append(report)
        common.write_summary(results_dir, reports, policy)
        common.write_progress(results_dir, reports, pending)

    common.write_summary(results_dir, reports, policy)

    if args.fail_on == "all":
        expected = [target.model_id for target in targets]
        actual = [report["model_id"] for report in reports]
        failing = [
            report
            for report in reports
            if report["overall_status"] != "PASSED"
        ]
        if actual != expected or failing:
            raise SystemExit(
                "Radeon Pod notebook CI was incomplete or not all-pass: "
                f"expected={len(expected)}, reported={len(actual)}, "
                f"not_passed={len(failing)}"
            )


def install_signal_handlers() -> None:
    def stop(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)


if __name__ == "__main__":
    install_signal_handlers()
    try:
        main()
    except KeyboardInterrupt as exc:
        print(f"Interrupted: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(130)
