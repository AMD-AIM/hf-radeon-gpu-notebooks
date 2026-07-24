from __future__ import annotations

import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
WORKFLOW = (
    REPO / ".github" / "workflows" / "huggingface-oneclick-notebook-ci.yml"
)
WORKFLOW_TEXT = WORKFLOW.read_text()


def step_block(name: str) -> str:
    marker = f"      - name: {name}\n"
    start = WORKFLOW_TEXT.index(marker)
    end = WORKFLOW_TEXT.find("\n      - name: ", start + len(marker))
    if end == -1:
        end = len(WORKFLOW_TEXT)
    return WORKFLOW_TEXT[start:end]


def step_script(name: str) -> str:
    lines = step_block(name).splitlines()
    run_index = next(
        index for index, line in enumerate(lines) if line.strip() == "run: |"
    )
    run_indent = len(lines[run_index]) - len(lines[run_index].lstrip())
    script_lines = []
    for line in lines[run_index + 1 :]:
        if line.strip() and len(line) - len(line.lstrip()) <= run_indent:
            break
        script_lines.append(line)
    return textwrap.dedent("\n".join(script_lines))


def run_step_script(
    name: str, workspace: Path, extra_environment: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["GITHUB_WORKSPACE"] = str(workspace)
    if extra_environment:
        environment.update(extra_environment)
    return subprocess.run(
        ["bash", "-e", "-c", step_script(name)],
        cwd=workspace,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


class WorkflowOutputIsolationTests(unittest.TestCase):
    def test_cleanup_only_removes_stale_results_before_checkout(self):
        self.assertLess(
            WORKFLOW_TEXT.index("      - name: Clear stale run outputs\n"),
            WORKFLOW_TEXT.index("      - name: Checkout target revision\n"),
        )

        with (
            tempfile.TemporaryDirectory() as directory,
            tempfile.TemporaryDirectory() as external_directory,
        ):
            workspace = Path(directory)
            results = workspace / "results"
            results.mkdir()
            (results / "summary.md").write_text("OLD RUN: 25 PASS\n")
            (results / "old-result.json").write_text("{}\n")
            sibling = workspace / "keep.txt"
            sibling.write_text("keep\n")

            run_step_script("Clear stale run outputs", workspace)

            self.assertFalse(results.exists())
            self.assertEqual(sibling.read_text(), "keep\n")

            external = Path(external_directory)
            external_marker = external / "must-survive.txt"
            external_marker.write_text("keep\n")
            results.symlink_to(external, target_is_directory=True)

            run_step_script("Clear stale run outputs", workspace)

            self.assertFalse(results.exists())
            self.assertEqual(external_marker.read_text(), "keep\n")

    def test_checkout_failure_publishes_honest_summary_not_stale_results(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            results = workspace / "results"
            results.mkdir()
            (results / "summary.md").write_text("OLD RUN: 25 PASS\n")
            github_summary = workspace / "github-step-summary.md"

            run_step_script("Clear stale run outputs", workspace)
            run_step_script(
                "Publish summary",
                workspace,
                {
                    "CHECKOUT_OUTCOME": "failure",
                    "GITHUB_STEP_SUMMARY": str(github_summary),
                },
            )

            summary = github_summary.read_text()
            self.assertIn("Infrastructure failure", summary)
            self.assertIn("repository checkout did not complete", summary)
            self.assertIn("Notebook execution did not start", summary)
            self.assertNotIn("OLD RUN: 25 PASS", summary)

    def test_current_run_summary_is_published_after_successful_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            results = workspace / "results"
            results.mkdir()
            current_summary = "CURRENT RUN: model results\n"
            (results / "summary.md").write_text(current_summary)
            github_summary = workspace / "github-step-summary.md"

            run_step_script(
                "Publish summary",
                workspace,
                {
                    "CHECKOUT_OUTCOME": "success",
                    "GITHUB_STEP_SUMMARY": str(github_summary),
                },
            )

            self.assertEqual(github_summary.read_text(), current_summary)


class WorkflowFailureHandlingTests(unittest.TestCase):
    def test_post_checkout_mutations_and_upload_require_checkout_success(self):
        checkout = step_block("Checkout target revision")
        sync = step_block("Sync downloaded notebook snapshots")
        upload = step_block("Upload results")

        self.assertIn("        id: checkout\n", checkout)
        checkout_guard = "steps.checkout.outcome == 'success'"
        self.assertIn(checkout_guard, sync)
        self.assertIn(checkout_guard, upload)


class WorkflowCacheConfigurationTests(unittest.TestCase):
    def test_hf_and_transformers_use_the_same_model_cache(self):
        self.assertIn(
            'HF_CACHE_ROOT="/tmp/huggingface_cache/'
            '${{ github.run_id }}-${{ github.run_attempt }}"',
            WORKFLOW_TEXT,
        )
        self.assertIn('-e HF_HUB_CACHE="$HF_CACHE_ROOT/hub"', WORKFLOW_TEXT)
        self.assertIn('-e TRANSFORMERS_CACHE="$HF_CACHE_ROOT/hub"', WORKFLOW_TEXT)
        self.assertNotIn(
            'TRANSFORMERS_CACHE="$HF_CACHE_ROOT/transformers"',
            WORKFLOW_TEXT,
        )


class WorkflowGitTransportTests(unittest.TestCase):
    def test_checkout_proxy_rewrite_is_step_scoped(self):
        checkout = step_block("Checkout target revision")
        job_prefix = WORKFLOW_TEXT[: WORKFLOW_TEXT.index(checkout)]

        self.assertIn('GIT_CONFIG_COUNT: "1"', checkout)
        self.assertIn(
            'GIT_CONFIG_KEY_0: "url.https://gh-test.anruicloud.com/.insteadOf"',
            checkout,
        )
        self.assertIn('GIT_CONFIG_VALUE_0: "https://github.com/"', checkout)
        self.assertIn("persist-credentials: true", checkout)
        self.assertNotIn("github-server-url:", checkout)
        self.assertNotIn("GIT_CONFIG_KEY_0:", job_prefix)

    def test_notebook_sync_pulls_via_proxy_but_pushes_directly(self):
        sync = step_block("Sync downloaded notebook snapshots")

        self.assertIn(
            '"url.https://gh-test.anruicloud.com/.insteadOf=https://github.com/"',
            sync,
        )
        self.assertIn(
            'retry_git git_via_fetch_proxy pull --rebase origin "$BRANCH"',
            sync,
        )
        self.assertIn('retry_git git push origin "HEAD:$BRANCH"', sync)
        self.assertNotIn("git_via_fetch_proxy push", sync)


if __name__ == "__main__":
    unittest.main()
