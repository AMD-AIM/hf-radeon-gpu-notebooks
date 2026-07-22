# Hugging Face Radeon GPU Notebook CI

This repository is dedicated to CI development for Hugging Face one-click
notebooks on AMD Radeon GPUs.

The active CI implementation, notebook sources, model list, and supporting
tools live on the
[`ci/huggingface_oneclick_workaround`](https://github.com/AMD-AIM/hf-radeon-gpu-notebooks/tree/ci/huggingface_oneclick_workaround)
branch.

GitHub-hosted scheduled dispatch is intentionally disabled. Run the workflow
manually from the **Actions** tab, or use:

```bash
gh workflow run huggingface-oneclick-notebook-ci.yml \
  --repo AMD-AIM/hf-radeon-gpu-notebooks \
  --ref ci/huggingface_oneclick_workaround \
  --field filter="" \
  --field use_runner_hf_cache=false
```

The default branch retains only this guide and a minimal `workflow_dispatch`
bridge because GitHub requires a manually dispatched workflow to exist on the
default branch.
