# Radeon Global notebook CI

This branch uses the Radeon Global One-Click backend as the notebook execution
environment. The self-hosted runner is only a long-running controller: it does
not select a local GPU, start Docker, mount a runner Hugging Face cache, or run
model code.

## Per-model lifecycle

For every enabled row in `doc/ci_target_models.csv`, the controller performs
these steps sequentially:

1. Download and normalize the native Hugging Face notebook. This preparation is
   excluded from model timing.
2. Create one Radeon One-Click Pod for the model and wait for its Jupyter API.
   Pod creation and readiness are excluded from model timing.
3. Upload the normalized notebook and create one Jupyter kernel session.
4. Execute the notebook's code cells in order through the Jupyter kernel
   WebSocket protocol.
5. Close the Jupyter session and delete the Pod. Session preparation and Pod
   deletion are excluded from model timing.

The Pod is deleted in a `finally` path. A mode-0600 ownership file records the
user, model, and instance id without storing the Radeon API token or Jupyter
access URL. The workflow has a second `if: always()` cleanup step for interrupted
runs, and refuses to delete a current instance whose id differs from the one it
created.

## Cell retry behavior

Jupyter kernels do not provide one atomic "run from this cell to the end"
request. JupyterLab implements that UI command by sending one
`execute_request` per cell, so the controller does the same.

If a cell raises an error or times out, the controller stays on that cell and
executes it up to three total times. Once it succeeds, execution continues with
the remaining cells. This matches manually rerunning a cell whose model
download was interrupted: partial files remain in the same Pod cache and the
original notebook cell initiates the next download attempt.

If the kernel process or WebSocket is lost, in-memory notebook state no longer
exists. The controller creates another session and replays from the first cell;
it allows at most three kernel sessions.

No separate `hf download` cell is injected. Consequently, model download time
and attempts cannot be separated reliably from notebook execution. The summary
shows `Download: in notebook`, `Model Download Tries: \`, and reports observable
cell retries instead.

## Timing

The timed interval starts immediately before the first normalized code cell is
sent to the kernel and ends when the last attempted code cell returns both its
execution reply and kernel `idle` status. The interval includes:

- model downloads performed by notebook cells;
- inference and other notebook work;
- failed cell attempts and retry delays;
- kernel reconstruction and replay after a kernel failure.

It excludes notebook retrieval/normalization, initial Jupyter upload/session
creation, Pod creation/readiness, result writing, HTML conversion, and Pod
deletion.

## GitHub configuration

Configure these values before dispatching the workflow:

- repository secret `RADEON_API_TOKEN`;
- repository variable `RADEON_USER_NAME`;
- repository secret `HF_ONECLICK_TOKEN`.

The workflow passes the Hugging Face token into the Pod environment. The
controller never writes the Radeon API token or Jupyter access URL to ownership
state or result metadata, and it recursively redacts known secrets from logs
and executed notebook values before producing artifacts. Notebook sources must
still not intentionally persist credentials.

The controller currently remains on the W7900 self-hosted runner because a
standard GitHub-hosted job has a six-hour execution limit, while all models run
sequentially and may exceed it. Moving the controller fully to GitHub-hosted
runners requires one `max-parallel: 1` matrix job per model plus a final result
aggregation job.

## Local test

Use an isolated environment and export credentials without writing them into
the repository:

```bash
python3 -m venv /tmp/hf-radeon-pod-controller
/tmp/hf-radeon-pod-controller/bin/python -m pip install \
  -r tools/requirements-radeon-pod-ci.txt

export RADEON_API_TOKEN='...'
export RADEON_USER_NAME='...'
export HF_TOKEN='...'
export HF_ENDPOINT='http://134.199.133.77'

/tmp/hf-radeon-pod-controller/bin/python -u \
  tools/run_radeon_pod_notebooks.py \
  --filter 'Qwen/Qwen3-0.6B' \
  --results-dir /tmp/hf-radeon-pod-results \
  --state-file /tmp/hf-radeon-pod-state.json
```

Before a real run, `GET /current` must report `not_found`. If a manually created
or otherwise unowned Pod already exists, the controller stops rather than
replacing it.
