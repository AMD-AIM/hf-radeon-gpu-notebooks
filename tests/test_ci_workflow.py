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


class WorkflowRadeonGlobalBackendTests(unittest.TestCase):
    def test_workflow_identity_matches_radeon_global_branch(self):
        self.assertTrue(
            WORKFLOW_TEXT.startswith(
                "name: HF One-Click CI\n"
                'run-name: "[Run] Radeon Global CI"\n'
            )
        )
        self.assertIn(
            "  push:\n    branches: [hf_oneclick_radeon_global]\n",
            WORKFLOW_TEXT,
        )

    def test_self_hosted_runner_is_only_a_long_running_pod_controller(self):
        execute = step_block("Execute notebook CI")

        self.assertIn("    runs-on: [self-hosted, rocm, w7900]", WORKFLOW_TEXT)
        self.assertIn("does not use this runner's GPU, Docker", WORKFLOW_TEXT)
        self.assertIn("tools/run_radeon_pod_notebooks.py", execute)
        self.assertIn("RADEON_API_TOKEN: ${{ secrets.RADEON_API_TOKEN }}", execute)
        self.assertIn("RADEON_USER_NAME: ${{ vars.RADEON_USER_NAME }}", execute)
        self.assertNotIn("docker run", WORKFLOW_TEXT)
        self.assertNotIn("select_idle_rocm_gpu.py", WORKFLOW_TEXT)
        self.assertNotIn("/dev/kfd", WORKFLOW_TEXT)
        self.assertNotIn("/disk/ssd2/huggingface_cache", WORKFLOW_TEXT)
        self.assertNotIn("use_runner_hf_cache", WORKFLOW_TEXT)

    def test_owned_pod_cleanup_runs_even_after_failure(self):
        cleanup = step_block("Ensure owned Radeon Pod is deleted")

        self.assertIn("        if: always()", cleanup)
        self.assertIn("--cleanup-state", cleanup)
        self.assertIn('--state-file "$RADEON_POD_STATE"', cleanup)
        self.assertNotIn("results/", cleanup)
        self.assertLess(
            WORKFLOW_TEXT.index("      - name: Ensure owned Radeon Pod is deleted\n"),
            WORKFLOW_TEXT.index("      - name: Sync downloaded notebook snapshots\n"),
        )

    def test_controller_dependencies_and_tests_run_before_notebooks(self):
        install = WORKFLOW_TEXT.index("      - name: Install controller dependencies\n")
        unit_tests = WORKFLOW_TEXT.index("      - name: Run controller unit tests\n")
        execute = WORKFLOW_TEXT.index("      - name: Execute notebook CI\n")

        self.assertLess(install, unit_tests)
        self.assertLess(unit_tests, execute)
        self.assertIn(
            "-r tools/requirements-radeon-pod-ci.txt",
            step_block("Install controller dependencies"),
        )
        self.assertIn(
            '>> "$GITHUB_ENV"',
            step_block("Install controller dependencies"),
        )

    def test_models_are_processed_strictly_serially(self):
        controller = (REPO / "tools" / "run_radeon_pod_notebooks.py").read_text()

        self.assertIn("    for target in targets:\n", controller)
        self.assertIn(
            "        report = run_one(target, args, results_dir, client)\n",
            controller,
        )
        self.assertNotIn("matrix:", WORKFLOW_TEXT)
        self.assertNotIn("strategy:", WORKFLOW_TEXT)


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
