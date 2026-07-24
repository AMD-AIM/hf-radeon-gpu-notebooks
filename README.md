# Hugging Face Radeon GPU Notebook CI

This repository is dedicated to CI development for Hugging Face one-click
notebooks on AMD Radeon GPUs.

The active CI implementations, notebook sources, model list, and supporting
tools live on these branches:

- [`hf_oneclick_local_machine`](https://github.com/AMD-AIM/hf-radeon-gpu-notebooks/tree/hf_oneclick_local_machine):
  executes notebooks on the W7900 self-hosted runner.
- [`hf_oneclick_radeon_global`](https://github.com/AMD-AIM/hf-radeon-gpu-notebooks/tree/hf_oneclick_radeon_global):
  executes notebooks in serially created Radeon Global One-Click Pods.

GitHub-hosted scheduled dispatch is intentionally disabled. Run the workflow
manually from the **Actions** tab, or use:

```bash
gh workflow run huggingface-oneclick-notebook-ci.yml \
  --repo AMD-AIM/hf-radeon-gpu-notebooks \
  --ref main \
  --field targets='["radeon-global","local-machine"]' \
  --field filter="" \
  --field use_runner_hf_cache=true
```

The default branch retains only this guide and a minimal `workflow_dispatch`
bridge because GitHub requires a manually dispatched workflow to exist on the
default branch. The `targets` input is an ordered JSON array. By default, the
bridge dispatches `hf_oneclick_radeon_global` first and
`hf_oneclick_local_machine` second. A single target or the reverse order is
also supported.
